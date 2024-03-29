# coding=utf-8
# Copyright 2022 Foivos Tsimpourlas.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Agents module for reinforcement learning.
"""
from cmath import inf
from code import interact
import pathlib
import typing
import tqdm
import numpy as np

from deeplearning.benchpress.reinforcement_learning import interactions
from deeplearning.benchpress.reinforcement_learning import model
from deeplearning.benchpress.reinforcement_learning import env
from deeplearning.benchpress.reinforcement_learning import hooks
from deeplearning.benchpress.reinforcement_learning.config import QValuesConfig
from deeplearning.benchpress.models import language_models
from deeplearning.benchpress.proto import reinforcement_learning_pb2
from deeplearning.benchpress.corpuses import tokenizers
from deeplearning.benchpress.util import pytorch
from deeplearning.benchpress.util import distrib
from deeplearning.benchpress.util import environment
from deeplearning.benchpress.util import logging as l

from absl import flags

FLAGS = flags.FLAGS

torch = pytorch.torch

class Policy(object):
  """
  The policy selected over Q-Values
  """
  def __init__(self, action_temp: float, token_temp: float):
    self.action_temperature = action_temp
    self.token_temperature  = token_temp
    return

  def SampleActions(self,
                    action_logits  : torch.FloatTensor,
                    actual_lengths : typing.Tuple[torch.LongTensor, torch.LongTensor],
                    ) -> typing.Tuple[int, int]:
    """
    Get the Q-Values for action and apply policy on it.
    """
    actions = torch.zeros((action_logits.shape[0]), dtype = torch.long)
    batch_idxs, seq_idxs = actual_lengths
    for bidx, sidx, seq_logits in zip(batch_idxs, seq_idxs, action_logits):
      try:
        ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
            temperature = self.action_temperature if self.action_temperature is not None else 1.0,
            logits = seq_logits[:(sidx * len(interactions.ACTION_TYPE_SPACE))],
            validate_args = False if "1.9." in torch.__version__ else None,
          ).sample()
        action = torch.argmax(ct, dim = -1)
        actions[bidx] = action
      except Exception as e:
        l.logger().error(seq_logits[:(sidx * len(interactions.ACTION_TYPE_SPACE))])
        raise e
    return actions

  def SampleTokens(self, token_logits: torch.FloatTensor) -> int:
    """
    Get logit predictions for token and apply policy on it.
    """
    ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = self.token_temperature if self.token_temperature is not None else 1.0,
        logits = token_logits,
        validate_args = False if "1.9." in torch.__version__ else None,
      ).sample()
    tokens = torch.argmax(ct, dim = -1)
    return tokens

class Agent(object):
  """
  Benchmark generation RL-Agent.
  """
  def __init__(self,
               config            : reinforcement_learning_pb2.RLModel,
               language_model    : language_models.Model,
               tokenizer         : tokenizers.TokenizerBase,
               feature_tokenizer : tokenizers.FeatureTokenizer,
               cache_path        : pathlib.Path
               ):

    self.cache_path = cache_path / "agent"
    self.ckpt_path  = self.cache_path / "checkpoint"
    self.log_path   = self.cache_path / "logs"
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
      self.ckpt_path.mkdir(exist_ok = True, parents = True)
      self.log_path.mkdir(exist_ok = True, parents = True)

    self.config            = config
    self.language_model    = language_model

    self.tokenizer         = tokenizer
    self.feature_tokenizer = feature_tokenizer
    self.qv_config = QValuesConfig.from_config(
      self.config,
      self.language_model.backend.config.architecture.max_position_embeddings,
      self.tokenizer,
      self.feature_tokenizer,
      self.language_model,
    )
    self.policy  = Policy(
      action_temp = self.qv_config.action_temperature,
      token_temp  = self.qv_config.token_temperature,
    )
    return

  def _ConfigModelParams(self, learning_rate: float) -> None:
    """
    Initialize torch models and send them to device.
    """
    self.action_actor  = model.ActionQV(self.language_model, self.qv_config).to(pytorch.device)
    self.action_critic = model.ActionQV(self.language_model, self.qv_config, is_critic = True).to(pytorch.device)
    self.token_actor   = model.ActionLanguageModelQV(self.language_model, self.qv_config).to(pytorch.device)
    self.token_critic  = model.ActionLanguageModelQV(self.language_model, self.qv_config, is_critic = True).to(pytorch.device)
    if pytorch.num_nodes > 1:
      self.action_actor = torch.nn.DistributedDataParallel(
        self.action_actor,
        device_ids    = [pytorch.offset_device],
        output_device = pytorch.offset_device,
        find_unused_parameters = True,
      )
      self.action_critic = torch.nn.DistributedDataParallel(
        self.action_critic,
        device_ids    = [pytorch.offset_device],
        output_device = pytorch.offset_device,
        find_unused_parameters = True,
      )
      self.token_actor = torch.nn.DistributedDataParallel(
        self.token_actor,
        device_ids    = [pytorch.offset_device],
        output_device = pytorch.offset_device,
        find_unused_parameters = True,
      )
      self.token_critic = torch.nn.DistributedDataParallel(
        self.token_critic,
        device_ids    = [pytorch.offset_device],
        output_device = pytorch.offset_device,
        find_unused_parameters = True,
      )
    elif pytorch.num_gpus > 1:
      self.action_actor  = torch.nn.DataParallel(self.action_actor)
      self.action_critic = torch.nn.DataParallel(self.action_critic)
      self.token_actor   = torch.nn.DataParallel(self.token_actor)
      self.token_critic  = torch.nn.DataParallel(self.token_critic)
    self.action_optim = torch.optim.Adam(
      list(self.action_actor.parameters()) + list(self.action_critic.parameters()),
      lr = learning_rate
    )
    self.token_optim  = torch.optim.Adam(
      list(self.token_actor.parameters()) + list(self.token_critic.parameters()),
      lr = learning_rate
    )
    return

  def Train(self,
            env               : env.Environment,
            num_epochs        : int,
            num_episodes      : int, # Equivalent to batch size
            steps_per_episode : int, # Depth length of single trajectory.
            num_updates       : int,
            gamma             : float,
            lr                : float,
            lam               : float,
            epsilon           : float,
            value_loss_coeff  : float,
            entropy_coeff     : float,
            ) -> None:
    """
    Run PPO over policy and train the agent.
    """
    self._ConfigModelParams(learning_rate = lr)
    self.ckpt_step = max(0, self.loadCheckpoint())


    ########### DOES LM WORK ALONE ?
    code = "[START][HOLE]kernel[END]"
    encoded = list(self.tokenizer.TokenizeString(code))
    encoded = encoded + [self.tokenizer.padToken] * (self.language_model.backend.config.architecture.max_position_embeddings - len(encoded))

    inputs = {
      'input_ids'    : torch.LongTensor(encoded).unsqueeze(0).to(pytorch.device),
      'input_mask'   : (torch.LongTensor(encoded) != self.tokenizer.padToken).unsqueeze(0).to(pytorch.device),
      'position_ids' : torch.arange(self.language_model.backend.config.architecture.max_position_embeddings).unsqueeze(0).to(pytorch.device),
      'mask_labels'  : None,
      'input_features': None,
    }

    out = self.language_model.backend.model_step(
      self.language_model.backend.GetEncoderModule(with_checkpoint = True, without_label_head = False).to(pytorch.device),
      inputs,
    )

    preds = torch.argmax(out['prediction_logits'], dim = -1)
    l.logger().info(self.tokenizer.tokensToString([int(x) for x in preds.squeeze(0)[:10].cpu()]))

    ########### DOES LM WORK ALONE ?



    if self.is_world_process_zero():
      rollout_hook = hooks.tensorMonitorHook(
        self.log_path,
        self.ckpt_step,
        1, 1,
        average = False,
      )
    #   train_hook   = hooks.tensorMonitorHook(
    #     self.logfile_path,
    #     self.current_step,
    #     min(self.steps_per_epoch, FLAGS.monitor_frequency)
    #   )

    action_type_distrib = {
      k: ([], []) for k in interactions.ACTION_TYPE_SPACE.keys()
    }
    index_type_distrib = {
      k: ([], []) for k in range(self.qv_config.max_position_embeddings)
    }

    for ep in range(num_epochs):
      # Run a batch of episodes.
      input_ids, final_state, masked_input_ids, feature_ids,\
      action_values, action_predictions, action_policy_probs,\
      token_values, token_predictions, token_policy_probs,\
      use_lm, rewards, discounted_rewards, done = self.rollout(
        env, num_episodes, steps_per_episode, gamma,
      )
      action_advantages, token_advantages = self.gae(
        rewards,
        action_values,
        token_values,
        use_lm,
        done,
        gamma,
        lam
      )
      # Compute reward-to-gos.
      action_reward_to_go = action_advantages + action_values.squeeze(-1)
      token_reward_to_go  = token_advantages  + token_values.squeeze(-1)

      # Nornmalize advantages.
      action_advantages = (action_advantages - action_advantages.mean()) / (action_advantages.std() + 1e-5)
      token_advantages  = (token_advantages - token_advantages.mean()) / (token_advantages.std() + 1e-5)

      # Set the batch size.
      batch_size  = int(input_ids.shape[0])
      num_batches = int(input_ids.shape[1])

      # Reshape to 2 dimensions.
      action_advantages   = torch.reshape(action_advantages,   (-1, ) + action_advantages.shape[2:])
      token_advantages    = torch.reshape(token_advantages,    (-1, ) + token_advantages.shape[2:])
      action_reward_to_go = torch.reshape(action_reward_to_go, (-1, ) + action_reward_to_go.shape[2:])
      token_reward_to_go  = torch.reshape(token_reward_to_go,  (-1, ) + token_reward_to_go.shape[2:])
      action_values       = torch.reshape(action_values,       (-1, ) + action_values.shape[2:])
      token_values        = torch.reshape(token_values,        (-1, ) + token_values.shape[2:])
      action_predictions  = torch.reshape(action_predictions,  (-1, ) + action_predictions.shape[2:])
      token_predictions   = torch.reshape(token_predictions,   (-1, ) + token_predictions.shape[2:])
      use_lm              = torch.reshape(use_lm,              (-1, ) + use_lm.shape[2:])
      input_ids           = torch.reshape(input_ids,           (-1, ) + input_ids.shape[2:])
      masked_input_ids    = torch.reshape(masked_input_ids,    (-1, ) + masked_input_ids.shape[2:])
      feature_ids         = torch.reshape(feature_ids,         (-1, ) + feature_ids.shape[2:])
      action_policy_probs = torch.reshape(action_policy_probs, (-1, ) + action_policy_probs.shape[2:])
      token_policy_probs  = torch.reshape(token_policy_probs,  (-1, ) + token_policy_probs.shape[2:])

      if environment.WORLD_SIZE > 1:
        raise NotImplementedError("Gather all the tensors here ?")

      for k in action_type_distrib.keys():
        action_type_distrib[k][0].append(ep)
        action_type_distrib[k][1].append(0)
      for k in index_type_distrib.keys():
        index_type_distrib[k][0].append(ep)
        index_type_distrib[k][1].append(0)
      for act in action_predictions:
        act_type  = int(act) % len(interactions.ACTION_TYPE_SPACE)
        act_index = int(act) // len(interactions.ACTION_TYPE_SPACE)
        try:
          action_type_distrib[interactions.ACTION_TYPE_MAP[act_type]][1][ep] += 1
          index_type_distrib[act_index][1][ep] += 1
        except IndexError as e:
          l.logger().error(act_type)
          l.logger().error(act_index)
          l.logger().info(act)
          l.logger().warn(action_type_distrib)
          l.logger().info(index_type_distrib)
          raise e
      from deeplearning.benchpress.util import plotter as plt
      plt.GrouppedBars(
        groups = action_type_distrib,
        plot_name = "Acts_per_rollout_step",
        path = self.log_path,
      )
      plt.GrouppedBars(
        groups = index_type_distrib,
        plot_name = "pos_index_per_rollout_step",
        path = self.log_path,
      )
      ## Print the full trajectory with the best reward.
      best_full_traj = torch.argmax(discounted_rewards[:,-1], dim = -1)
      l.logger().info("Best full-trajectory sample:")
      print(self.tokenizer.tokensToString([int(x) for x in final_state[int(best_full_traj)]], ignore_token=self.tokenizer.padToken))


      # Split the data into batches in the num_workers dimension
      for epoch in tqdm.tqdm(range(num_updates), total = num_updates, desc = "Epoch"):
        for batch in tqdm.tqdm(range(num_batches), total = num_batches, desc = "Batch", leave = False):
          start = batch * batch_size
          end = (batch + 1) * batch_size
          # Step batch
          mean_action_loss, mean_token_loss = self.ppo_train_step(
            epsilon,
            value_loss_coeff,
            entropy_coeff,
            action_advantages   [start:end].to(pytorch.device),
            token_advantages    [start:end].to(pytorch.device),
            action_reward_to_go [start:end].to(pytorch.device),
            token_reward_to_go  [start:end].to(pytorch.device),
            action_values       [start:end].to(pytorch.device),
            token_values        [start:end].to(pytorch.device),
            action_predictions  [start:end],
            token_predictions   [start:end],
            use_lm              [start:end],
            input_ids           [start:end],
            masked_input_ids    [start:end],
            feature_ids         [start:end],
            action_policy_probs [start:end].to(pytorch.device),
            token_policy_probs  [start:end].to(pytorch.device),
          )
        # Probably here save the necessary checkpoints.
        # Also log the following stuff:
        # Rewards, advantages (?), size of code ?, rtg ? Distribution of actions selected ?
        # self.saveCheckpoint()
        if self.is_world_process_zero():
          rollout_hook.step(
            mean_action_loss  = float(mean_action_loss),
            mean_token_loss   = float(mean_token_loss),
            mean_final_reward = float(torch.mean(discounted_rewards[:,-1])),
          )
        self.ckpt_step += 1

        ## distribution of actions per 
    return

  def ppo_train_step(self,
                     epsilon             : float,
                     value_loss_coeff    : float,
                     entropy_coeff       : float,
                     action_advantages   : torch.FloatTensor,
                     token_advantages    : torch.FloatTensor,
                     action_reward_to_go : torch.FloatTensor,
                     token_reward_to_go  : torch.FloatTensor,
                     action_values       : torch.FloatTensor,
                     token_values        : torch.FloatTensor,
                     action_predictions  : torch.LongTensor,
                     token_predictions   : torch.LongTensor,
                     use_lm              : torch.BoolTensor,
                     input_ids           : torch.LongTensor,
                     masked_input_ids    : torch.LongTensor,
                     feature_ids         : torch.LongTensor,
                     action_policy_probs : torch.FloatTensor,
                     token_policy_probs  : torch.FloatTensor,
                     ) -> typing.Tuple[float, float]:
    """
    Run a batch through PPO training.
    Inputs:
      action_optim:
        Adam optimizer that handles action actor and critic.
      token_optim:
        Adam optimizer that handles token actor and critic.
      action_advantages:
        Calculated advantages for action model.
      token_advantages:
        Calculated advantages for token model.
      action_reward_to_go:
        Aggregated rewards for actions trajectory.
      token_reward_to_go:
        Aggregated rewards for tokens trajectory.
      action_values:
        Predicted values by action critic.
      token_values:
        Predicted values by token critic.
      action_predictions:
        Predicted action labels by action actor.
      token_predictions:
        Predicted token labels by token actor.
      use_lm:
        Indices of states that used the language model.
      input_ids:
        Input code for the action model.
      masked_input_ids:
        Masked input code for the token model. Contains masked code where use_lm==True, zeros otherwise.
      feature_ids:
        Tokenized vector of target state features.
      action_policy_probs:
        Predicted action label's probability.
      token_policy_probs:
        Predicted token label's probability.
    """
    # Enable training mode for these little fuckers.
    self.action_actor.train()
    self.action_critic.train()
    self.token_actor.train()
    self.token_critic.train()

    self.action_optim.zero_grad()
    self.token_optim.zero_grad()

    seq_len, feat_seq_len, batch_size = input_ids.shape[-1], feature_ids.shape[-1], input_ids.shape[0]

    mean_action_loss, action_backwards = 0.0, 0
    mean_token_loss,  token_backwards  = 0.0, 0

    # Prepare model inputs.
    feature_mask = feature_ids != self.feature_tokenizer.padToken
    feature_pos  = torch.arange(feat_seq_len, dtype = torch.long).repeat(batch_size, 1)
    input_mask   = feature_ids != self.feature_tokenizer.padToken
    input_pos    = torch.arange(seq_len, dtype = torch.long).repeat(batch_size, 1)

    # Run the batch again in actor/critic.
    # Actor model returns logits of action.
    action_actor_out = self.action_actor(
      encoder_feature_ids  = feature_ids.to(pytorch.device),
      encoder_feature_mask = feature_mask.to(pytorch.device),
      encoder_position_ids = feature_pos.to(pytorch.device),
      decoder_input_ids    = input_ids.to(pytorch.device),
      decoder_input_mask   = input_mask.to(pytorch.device),
      decoder_position_ids = input_pos.to(pytorch.device),
    )
    new_action_logits, new_action_probs = action_actor_out['action_logits'], action_actor_out['action_probs']
    # Critic model returns value logit.
    action_critic_out = self.action_critic(
      encoder_feature_ids  = feature_ids.to(pytorch.device),
      encoder_feature_mask = feature_mask.to(pytorch.device),
      encoder_position_ids = feature_pos.to(pytorch.device),
      decoder_input_ids    = input_ids.to(pytorch.device),
      decoder_input_mask   = input_mask.to(pytorch.device),
      decoder_position_ids = input_pos.to(pytorch.device),
    )
    new_action_values, new_action_values_probs = action_critic_out['action_logits'], action_critic_out['action_probs']
    # Sample the most likely action.
    actual_lengths = torch.where(input_ids == self.tokenizer.endToken)
    step_actions   = self.policy.SampleActions(new_action_logits, actual_lengths)
    # Collect the probability of said selected action, per episode.
    new_action_probs = new_action_probs[(torch.arange(new_action_probs.shape[0]), step_actions)]
    # Compute entropy of actions
    new_action_entropy = torch.distributions.categorical.Categorical(logits = new_action_logits).entropy()
    # Flatten the critic values.
    new_action_values = new_action_values.flatten()

    # Compute the PPO loss
    action_prob_ratio = torch.exp(new_action_probs) / torch.exp(action_policy_probs)
    a = action_prob_ratio * action_advantages
    b = torch.clamp(action_prob_ratio, 1 - epsilon, 1 + epsilon) * action_advantages
    action_ppo_loss = -1 * torch.mean(torch.min(a, b))

    # Compute the value function loss
    # Clipped loss - same idea as PPO loss, don't allow value to move too
    # far from where it was previously
    value_pred_clipped = action_values + (new_action_values - action_values).clamp(-epsilon, epsilon)
    value_losses = (new_action_values - action_reward_to_go) ** 2
    value_losses_clipped = (value_pred_clipped - action_reward_to_go) ** 2
    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)

    action_value_loss = value_loss.mean()
    action_entropy_loss = torch.mean(new_action_entropy)

    # Compute the final loss and backward.
    action_loss = action_ppo_loss + value_loss_coeff * action_value_loss - entropy_coeff * action_entropy_loss
    action_loss.backward()
    mean_action_loss += action_loss.item()
    action_backwards += 1

    torch.nn.utils.clip_grad_norm_(self.action_actor.parameters(), .5)
    torch.nn.utils.clip_grad_norm_(self.action_critic.parameters(), .5)
    self.action_optim.step()

    if torch.any(use_lm):
      # Get the indices where use_lm is True.
      lm_indices = torch.where(use_lm == True)[0]
      # Prepare token model inputs.
      lm_feature_ids  = torch.index_select(feature_ids, 0, lm_indices)
      lm_feature_mask = lm_feature_ids != self.feature_tokenizer.padToken
      lm_feat_pos_id  = torch.arange(feat_seq_len, dtype = torch.long).repeat(lm_feature_ids.shape[0], 1)
      lm_input_ids    = torch.index_select(masked_input_ids, 0, lm_indices)
      lm_input_mask   = lm_input_ids != self.tokenizer.padToken
      lm_pos_id       = torch.arange(seq_len, dtype = torch.long).repeat(lm_input_ids.shape[0], 1)

      # Keep track of where [HOLE] reside.
      ep_idx, seq_idx = torch.where(lm_input_ids == self.tokenizer.holeToken)

      # Run the batch in actor/critic.
      # The input indices are based on those the rollout action actor decided to use the LM.
      # We directly use masked_input_ids for this reason.
      # Actor model returns logits of the token predictions.
      token_actor_out = self.token_actor(
        encoder_feature_ids  = lm_feature_ids.to(pytorch.device),
        encoder_feature_mask = lm_feature_mask.to(pytorch.device),
        encoder_position_ids = lm_feat_pos_id.to(pytorch.device),
        decoder_input_ids    = lm_input_ids.to(pytorch.device),
        decoder_input_mask   = lm_input_mask.to(pytorch.device),
        decoder_position_ids = lm_pos_id.to(pytorch.device),
      )
      t, new_token_probs = token_actor_out['token_logits'], token_actor_out['token_probs']
      # Collect the logits but only for the hole indices.
      new_token_logits = t[(ep_idx, seq_idx)]
      new_token_probs  = new_token_probs[(ep_idx, seq_idx)]
      # Critic model returns value logit.
      token_critic_out = self.token_critic(
        encoder_feature_ids  = lm_feature_ids.to(pytorch.device),
        encoder_feature_mask = lm_feature_mask.to(pytorch.device),
        encoder_position_ids = lm_feat_pos_id.to(pytorch.device),
        decoder_input_ids    = lm_input_ids.to(pytorch.device),
        decoder_input_mask   = lm_input_mask.to(pytorch.device),
        decoder_position_ids = lm_pos_id.to(pytorch.device),
      )
      new_token_values, new_token_values_probs = token_critic_out['token_logits'], token_critic_out['token_probs']
      # Collect the critic's value for this hole index.
      new_token_values       = new_token_values[(ep_idx, seq_idx)]
      new_token_values_probs = new_token_values_probs[(ep_idx, seq_idx)]
      # According to policy, select the best token.
      new_tokens        = self.policy.SampleTokens(new_token_logits)
      # Get probability of said token, per sequence.
      new_token_probs   = new_token_probs[(torch.arange(new_token_probs.shape[0]), new_tokens)]
      # Calculate the entropy of new token logits.
      new_token_entropy = torch.distributions.categorical.Categorical(logits = new_token_logits).entropy()
      # Flatten critic values.
      new_token_values  = new_token_values.flatten()

      # Keep only the advantages and policy probs for the indices where the LM was used.
      lm_indices         = lm_indices.to(pytorch.device)
      token_advantages   = torch.index_select(token_advantages,   0, lm_indices)
      token_reward_to_go = torch.index_select(token_reward_to_go, 0, lm_indices)
      token_policy_probs = torch.index_select(token_policy_probs, 0, lm_indices)
      token_values       = torch.index_select(token_values,       0, lm_indices)

      # Compute the PPO loss
      token_prob_ratio = torch.exp(new_token_probs) / torch.exp(token_policy_probs.to(pytorch.device))
      a = token_prob_ratio * token_advantages.to(pytorch.device)
      b = torch.clamp(token_prob_ratio, 1 - epsilon, 1 + epsilon) * token_advantages
      token_ppo_loss = -1 * torch.mean(torch.min(a, b))

      # Compute the value function loss
      # Clipped loss - same idea as PPO loss, don't allow value to move too
      # far from where it was previously
      value_pred_clipped = token_values + (new_token_values - token_values).clamp(-epsilon, epsilon)
      value_losses = (new_token_values - token_reward_to_go) ** 2
      value_losses_clipped = (value_pred_clipped - token_reward_to_go) ** 2
      value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)

      token_value_loss = value_loss.mean()
      token_entropy_loss = torch.mean(new_token_entropy)

      # Compute the final loss and backward.
      token_loss = token_ppo_loss + value_loss_coeff * token_value_loss - entropy_coeff * token_entropy_loss
      token_loss.backward()
      mean_token_loss += token_loss.item()
      token_backwards += 1

      torch.nn.utils.clip_grad_norm_(self.token_actor.parameters(), .5)
      torch.nn.utils.clip_grad_norm_(self.token_critic.parameters(), .5)
      self.token_optim.step()
    try:
      mean_action_loss = mean_action_loss / action_backwards
    except ZeroDivisionError:
      mean_action_loss = 0.0
    try:
      mean_token_loss = mean_token_loss / token_backwards
    except ZeroDivisionError:
      mean_token_loss = 0.0
    return mean_action_loss, mean_token_loss

  def rollout(self,
              env               : env.Environment,
              num_episodes      : int,
              steps_per_episode : int,
              gamma             : float,
              ) -> typing.Tuple[torch.Tensor]:
    """
    1. Initialize all tensors [(num_episodes x batch_size?) x steps_per_episode x state_tensor_size]

    2. for step in steps_per_episode:
      a) slice state tensor
      b) slice action tensor
      c) Pass through model
      d) env.step and assign new state to state tensor.
      e) Compute rewards and rtgs.
    """
    ## Reset the environment.
    state = env.reset()
    self.action_actor.eval()
    self.action_critic.eval()
    self.token_actor.eval()
    self.token_critic.eval()
    seq_len, feat_seq_len = len(state.encoded_code), len(state.encoded_features)
    ## Create state and action tensors.
    # State workload inputs.
    batch_feature_ids      = torch.LongTensor(state.encoded_features).unsqueeze(0).unsqueeze(0).repeat(num_episodes, steps_per_episode, 1) # Input features for workload
    batch_input_ids        = torch.zeros((num_episodes, steps_per_episode, seq_len), dtype = torch.long)  # Input code for workload
    batch_input_ids[:, 0]  = torch.LongTensor(state.encoded_code)                                         # Initialization of empty code for all episode's starting point of trajectory.
    batch_masked_input_ids = torch.zeros((num_episodes, steps_per_episode, seq_len), dtype = torch.long)  # Initialization of masked input ids tensor for token model.
    final_state            = torch.zeros((num_episodes, seq_len), dtype = torch.long)                     # The final state of all trajectories.
    # Action, token predictions and probs, critic values.
    action_predictions     = torch.zeros((num_episodes, steps_per_episode, 1), dtype = torch.long)        # All action predictions per episode, per state.
    action_policy_probs    = torch.zeros((num_episodes, steps_per_episode, 1), dtype = torch.float32)     # Probs of all actions predicted.
    action_values          = torch.zeros((num_episodes, steps_per_episode, 1), dtype = torch.float32)     # Values from critic for actions.
    token_predictions      = torch.zeros((num_episodes, steps_per_episode, 1), dtype = torch.long)        # All token predictions per episode, per state.
    token_policy_probs     = torch.zeros((num_episodes, steps_per_episode, 1), dtype = torch.float32)     # Probs of all tokens predicted.
    token_values           = torch.zeros((num_episodes, steps_per_episode, 1), dtype = torch.float32)     # Values from critic for tokens.
    use_lm                 = torch.zeros((num_episodes, steps_per_episode), dtype = torch.bool)           # Indices where LM was indeed used (action was 'add' or 'replace')
    ## Reward placeholders.
    rewards                = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)        # Rewards per episode, per action.
    discounted_rewards     = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)        # The aggregated-discounted rewards as the trajectories proceed.
    traj_disc_rewards      = torch.zeros((num_episodes), dtype = torch.float32)                           # The latest aggregated discounted reward computed.
    feature_dists          = torch.full((num_episodes,), -1, dtype = torch.float32)                        # A tensor with the last updated euclidean distance from feature target.
    done                   = torch.zeros((num_episodes, steps_per_episode), dtype = torch.bool)           # Done boolean tensor.

    ## Run execution loop.
    for step in tqdm.tqdm(range(steps_per_episode), total = steps_per_episode, desc = "Rollout {} episodes".format(num_episodes)):
      ## This loop unfolds all batch_size trajectories.
      # Input tensors
      feature_ids  = batch_feature_ids[:, step]
      feature_mask = feature_ids != self.feature_tokenizer.padToken
      feature_pos  = torch.arange(feat_seq_len, dtype = torch.long).repeat(feature_ids.shape[0], 1)
      input_ids    = batch_input_ids[:, step]
      input_mask   = input_ids != self.tokenizer.padToken
      input_pos    = torch.arange(seq_len, dtype = torch.long).repeat(input_ids.shape[0], 1)

      # Actor model returns logits of action.
      step_action_actor_out = self.action_actor(
        encoder_feature_ids  = feature_ids.to(pytorch.device),
        encoder_feature_mask = feature_mask.to(pytorch.device),
        encoder_position_ids = feature_pos.to(pytorch.device),
        decoder_input_ids    = input_ids.to(pytorch.device),
        decoder_input_mask   = input_mask.to(pytorch.device),
        decoder_position_ids = input_pos.to(pytorch.device),
      )
      step_action_logits, step_action_probs = step_action_actor_out['action_logits'], step_action_actor_out['action_probs']
      # Critic model returns value logit.
      step_action_critic_out = self.action_critic(
        encoder_feature_ids  = feature_ids.to(pytorch.device),
        encoder_feature_mask = feature_mask.to(pytorch.device),
        encoder_position_ids = feature_pos.to(pytorch.device),
        decoder_input_ids    = input_ids.to(pytorch.device),
        decoder_input_mask   = input_mask.to(pytorch.device),
        decoder_position_ids = input_pos.to(pytorch.device),
      )
      step_action_values, step_action_values_probs = step_action_critic_out['action_logits'], step_action_critic_out['action_probs']
      # Sample the most likely action.
      actual_lengths = torch.where(input_ids == self.tokenizer.endToken)
      step_actions   = self.policy.SampleActions(step_action_logits, actual_lengths)
      # Collect the probability of said selected action, per episode.
      step_action_probs = step_action_probs[(torch.arange(step_action_probs.shape[0]), step_actions)]

      # Declare here the augmented token vectors.
      augmented_step_token_values = torch.zeros((num_episodes, 1), dtype = torch.float32)
      augmented_step_tokens       = torch.zeros((num_episodes, 1), dtype = torch.long)
      augmented_step_token_probs  = torch.zeros((num_episodes, 1), dtype = torch.float32)

      ## Find which sequences need to sample a token.
      step_use_lm, masked_input_ids = env.intermediate_step(input_ids, step_actions)
      if torch.any(step_use_lm):
        ## If the language model needs to be invoked ('add' or 'replace')
        ## Fix the necessary batch of elements here.
        # Indices of starting tensors that need the LM.
        lm_indices = torch.where(step_use_lm == True)[0]

        # Input tensors.
        lm_feature_ids   = torch.index_select(feature_ids, 0, lm_indices)
        lm_feature_mask  = lm_feature_ids != self.feature_tokenizer.padToken
        lm_feat_pos_ids  = torch.arange(feat_seq_len, dtype = torch.long).repeat(lm_feature_ids.shape[0], 1)
        lm_input_ids     = torch.index_select(masked_input_ids, 0, lm_indices)
        lm_input_mask    = lm_input_ids != self.tokenizer.padToken
        lm_input_pos_ids = torch.arange(seq_len, dtype = torch.long).repeat(lm_input_ids.shape[0], 1)

        # Keep the hole indices to dereference the prediction logits.
        ep_idx, seq_idx = torch.where(lm_input_ids == self.tokenizer.holeToken)
        # Run the token actor, get token logits.
        step_token_actor_out = self.token_actor(
          encoder_feature_ids  = lm_feature_ids.to(pytorch.device),
          encoder_feature_mask = lm_feature_mask.to(pytorch.device),
          encoder_position_ids = lm_feat_pos_ids.to(pytorch.device),
          decoder_input_ids    = lm_input_ids.to(pytorch.device),
          decoder_input_mask   = lm_input_mask.to(pytorch.device),
          decoder_position_ids = lm_input_pos_ids.to(pytorch.device),
        )
        step_token_logits, step_token_probs = step_token_actor_out['token_logits'], step_token_actor_out['token_probs']
        # Keep the prediction scores only for the masked token.
        step_token_logits = step_token_logits[(ep_idx, seq_idx)]
        step_token_probs  = step_token_probs[(ep_idx, seq_idx)]
        # Collect value logit from critic.
        step_token_critic_out = self.token_critic(
          encoder_feature_ids  = lm_feature_ids.to(pytorch.device),
          encoder_feature_mask = lm_feature_mask.to(pytorch.device),
          encoder_position_ids = lm_feat_pos_ids.to(pytorch.device),
          decoder_input_ids    = lm_input_ids.to(pytorch.device),
          decoder_input_mask   = lm_input_mask.to(pytorch.device),
          decoder_position_ids = lm_input_pos_ids.to(pytorch.device),
        )
        step_token_values, step_token_values_probs = step_token_critic_out['token_logits'], step_token_critic_out['token_probs']
        # Get the critic's value only for masked index.
        step_token_values       = step_token_values[(ep_idx, seq_idx)]
        step_token_values_probs = step_token_values_probs[(ep_idx, seq_idx)]
        # According to policy, select the best token.
        step_tokens = self.policy.SampleTokens(step_token_logits)

        for inp in lm_input_ids:
          l.logger().info(self.tokenizer.tokensToString([int(x) for x in inp], ignore_token = self.tokenizer.padToken))
        for preds in step_tokens:
          l.logger().info(self.tokenizer.tokensToString([int(preds)], ignore_token = self.tokenizer.padToken))
        input()
        # Get probability of said token, per episode.
        step_token_probs = step_token_probs[(torch.arange(step_token_probs.shape[0]), step_tokens)]

        # First extend to original dimensions.
        # Store the modified - with token LM - codes to the original tensors.
        for nidx, lm_idx in zip(range(step_tokens.shape[0]), lm_indices):
          augmented_step_token_values[lm_idx] = step_token_values[nidx]
          augmented_step_tokens[lm_idx]       = step_tokens[nidx]
          augmented_step_token_probs[lm_idx]  = step_token_probs[nidx]

        # Here is the appropriate storing back.
        batch_masked_input_ids[:, step] = masked_input_ids
        token_values          [:, step] = augmented_step_token_values.detach().cpu()
        token_predictions     [:, step] = augmented_step_tokens.detach().cpu()
        token_policy_probs    [:, step] = augmented_step_token_probs.detach().cpu()

      ## Step environment and compute rewards.
      input_ids, reward, discounted_reward, d, step_use_lm = env.new_step(
        input_ids,
        step_actions,
        augmented_step_tokens,
        traj_disc_rewards,
        feature_dists,
        step_use_lm,
        gamma
      )
      ## Save data to rollout buffers.
      if step < steps_per_episode - 1:
        batch_input_ids  [:, step+1] = input_ids
      else:
        final_state                  = input_ids
      action_values      [:, step]   = step_action_values.detach().cpu()
      action_predictions [:, step]   = step_actions.unsqueeze(0).reshape((-1, 1)).detach().cpu()
      action_policy_probs[:, step]   = step_action_probs.unsqueeze(0).reshape((-1, 1)).detach().cpu()
      use_lm             [:, step]   = step_use_lm
      rewards            [:, step]   = reward
      traj_disc_rewards              = discounted_reward
      discounted_rewards [:, step]   = traj_disc_rewards
      done               [:, step]   = d
    return (
      batch_input_ids,        # source code states.
      final_state,            # The state of the trajectory after the last applied action.
      batch_masked_input_ids, # Masked source code for the language model.
      batch_feature_ids,      # Target feature vector state.
      action_values,          # Critic action logits.
      action_predictions,     # Actor sampled label actions.
      action_policy_probs,    # Actor probabilities of sampled actions.
      token_values,           # Critic token values.
      token_predictions,      # Actor sampled label tokens.
      token_policy_probs,     # Actor probabilities of sampled tokens.
      use_lm,                 # Indices of actions that  required language model.
      rewards,                # Rewards of each step.
      discounted_rewards,     # Discounted rewards of each step.
      done,                   # Whether this step concludes the episode.
    )

  def gae(self, rewards, action_values, token_values, use_lm, episode_ends, gamma, lam):
    """
    Compute generalized advantage estimate.
      rewards: a list of rewards at each step.
      values: the value estimate of the state at each step.
      episode_ends: an array of the same shape as rewards, with a 1 if the
          episode ended at that step and a 0 otherwise.
      gamma: the discount factor.
      lam: the GAE lambda parameter.
    """
    # Invert episode_ends to have 0 if the episode ended and 1 otherwise
    episode_ends = (episode_ends * -1) + 1
    action_values = action_values.squeeze(-1)
    token_values  = token_values.squeeze(-1)
    N = rewards.shape[0]
    T = rewards.shape[1]
    action_gae_step = torch.zeros((N, ))
    token_gae_step = torch.zeros((N, ))
    action_advantages = torch.zeros((N, T))
    token_advantages  = torch.zeros((N, T))
    for t in reversed(range(T - 1)):
      # First compute delta, which is the one-step TD error
      action_delta = rewards[:, t] + gamma * action_values[:, t + 1] * episode_ends[:, t] - action_values[:, t]
      token_delta  = rewards[:, t] + gamma * token_values[:, t + 1]  * episode_ends[:, t] - token_values[:, t]
      # Then compute the current step's GAE by discounting the previous step
      # of GAE, resetting it to zero if the episode ended, and adding this
      # step's delta
      # And store it
      action_gae_step = action_delta + gamma * lam * episode_ends[:, t] * action_gae_step
      token_gae_step  = token_delta + gamma * lam * episode_ends[:, t] * token_gae_step
      action_advantages[:, t] = action_delta + gamma * lam * episode_ends[:, t] * action_gae_step
      token_advantages[:, t]  = token_delta  + gamma * lam * episode_ends[:, t] * token_gae_step
    return action_advantages, token_advantages

  def saveCheckpoint(self) -> None:
    """
    Save agent state.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda prefix, x: self.ckpt_path / "{}{}_model-{}.pt".format(prefix, x, self.ckpt_step)
      if pytorch.torch_tpu_available:
        if pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
          pytorch.torch_xla_model.save(self.action_actor, ckpt_comp("actor", "action"))
          pytorch.torch_xla_model.save(self.action_critic, ckpt_comp("critic", "action"))
          pytorch.torch_xla_model.save(self.action_optim, ckpt_comp("action", "optimizer"))
          pytorch.torch_xla_model.save(self.token_optim, ckpt_comp("token", "optimizer"))
        pytorch.torch_xla.rendezvous("saving_optimizer_states")
      else:
        if isinstance(self.action_actor, torch.nn.DataParallel):
          torch.save(self.action_actor.module.state_dict(), ckpt_comp("actor", "action"))
        else:
          torch.save(self.action_actor.state_dict(), ckpt_comp("action", "action"))
        if isinstance(self.action_critic, torch.nn.DataParallel):
          torch.save(self.action_critic.module.state_dict(), ckpt_comp("critic", "action"))
        else:
          torch.save(self.action_critic.state_dict(), ckpt_comp("critic", "action"))
        if isinstance(self.token_actor, torch.nn.DataParallel):
          torch.save(self.token_actor.module.state_dict(), ckpt_comp("actor", "token"))
        else:
          torch.save(self.token_actor.state_dict(), ckpt_comp("action", "token"))
        if isinstance(self.token_critic, torch.nn.DataParallel):
          torch.save(self.token_critic.module.state_dict(), ckpt_comp("critic", "token"))
        else:
          torch.save(self.token_critic.state_dict(), ckpt_comp("critic", "token"))
        torch.save(self.action_optim.state_dict(), ckpt_comp("action", "optimizer"))
        torch.save(self.token_optim.state_dict(), ckpt_comp("token", "optimizer"))

      with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
        mf.write("train_step: {}\n".format(self.ckpt_step))
    self.ckpt_step += 1
    distrib.barrier()
    return
  
  def loadCheckpoint(self) -> None:
    """
    Load agent state.
    """
    if not (self.ckpt_path / "checkpoint.meta").exists():
      return -1
    with open(self.ckpt_path / "checkpoint.meta", 'w') as mf:
      get_step = lambda x: int(x.replace("\n", "").replace("train_step: ", ""))
      lines = mf.readlines()
      entries = set({get_step(x) for x in lines})

    ckpt_step = max(entries)
    ckpt_comp = lambda prefix, x: self.ckpt_path / "{}{}_model-{}.pt".format(prefix, x, ckpt_step)

    if isinstance(self.action_actor, torch.nn.DataParallel):
      try:
        self.action_actor.module.load_state_dict(torch.load(ckpt_comp("actor", "action")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("actor", "action")).items():
          if k[:7] == "module.":
            name = k[7:]
          else:
            name = "module." + k
          new_state_dict[name] = k
        self.action_actor.module.load_state_dict(new_state_dict)
    else:
      try:
        self.action_actor.module.load_state_dict(torch.load(ckpt_comp("actor", "action")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("actor", "action")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.action_actor.load_state_dict(new_state_dict)


    if isinstance(self.action_critic, torch.nn.DataParallel):
      try:
        self.action_critic.module.load_state_dict(torch.load(ckpt_comp("actor", "critic")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("actor", "critic")).items():
          if k[:7] == "module.":
            name = k[7:]
          else:
            name = "module." + k
          new_state_dict[name] = k
        self.action_critic.module.load_state_dict(new_state_dict)
    else:
      try:
        self.action_critic.module.load_state_dict(torch.load(ckpt_comp("actor", "critic")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("actor", "critic")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.action_critic.load_state_dict(new_state_dict)


    if isinstance(self.token_actor, torch.nn.DataParallel):
      try:
        self.token_actor.module.load_state_dict(torch.load(ckpt_comp("token", "action")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("token", "action")).items():
          if k[:7] == "module.":
            name = k[7:]
          else:
            name = "module." + k
          new_state_dict[name] = k
        self.token_actor.module.load_state_dict(new_state_dict)
    else:
      try:
        self.token_actor.module.load_state_dict(torch.load(ckpt_comp("token", "action")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("token", "action")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.token_actor.load_state_dict(new_state_dict)


    if isinstance(self.token_critic, torch.nn.DataParallel):
      try:
        self.token_critic.module.load_state_dict(torch.load(ckpt_comp("token", "critic")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("token", "critic")).items():
          if k[:7] == "module.":
            name = k[7:]
          else:
            name = "module." + k
          new_state_dict[name] = k
        self.token_critic.module.load_state_dict(new_state_dict)
    else:
      try:
        self.token_critic.module.load_state_dict(torch.load(ckpt_comp("token", "critic")))
      except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in torch.load(ckpt_comp("token", "critic")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.token_critic.load_state_dict(new_state_dict)

    if self.action_optim is not None and self.token_optim is not None and ckpt_step > 0:
      self.action_optim.load_state_dict(
        torch.load(ckpt_comp("action", "optimizer"), map_location = pytorch.device)
      )
      self.token_optim.load_state_dict(
        torch.load(ckpt_comp("token", "optimizer"), map_location = pytorch.device)
      )

    self.action_actor.eval()
    self.action_critic.eval()
    self.token_actor.eval()
    self.token_critic.eval()
    return ckpt_step

  def is_world_process_zero(self) -> bool:
    """
    Whether or not this process is the global main process (when training in a distributed fashion on
    several machines, this is only going to be :obj:`True` for one process).
    """
    if pytorch.torch_tpu_available:
      return pytorch.torch_xla_model.is_master_ordinal(local=False)
    elif pytorch.num_nodes > 1:
      return torch.distributed.get_rank() == 0
    else:
      return True
