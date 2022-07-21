"""
Agents module for reinforcement learning.
"""
from code import interact
import pathlib
import typing
import tqdm
import numpy as np

from deeplearning.clgen.reinforcement_learning import interactions
from deeplearning.clgen.reinforcement_learning import model
from deeplearning.clgen.reinforcement_learning import env
from deeplearning.clgen.reinforcement_learning.config import QValuesConfig
from deeplearning.clgen.models import language_models
from deeplearning.clgen.proto import reinforcement_learning_pb2
from deeplearning.clgen.corpuses import tokenizers
from deeplearning.clgen.util import pytorch
from deeplearning.clgen.util import environment
from deeplearning.clgen.util import logging as l

torch = pytorch.torch

class Policy(object):
  """
  The policy selected over Q-Values
  """
  def __init__(self, action_temp: float, token_temp: float):
    self.action_temperature = action_temp
    self.token_temperature  = token_temp
    return

  def SampleActions(self, action_logits: torch.FloatTensor) -> typing.Tuple[int, int]:
    """
    Get the Q-Values for action and apply policy on it.
    """
    ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = self.action_temperature if self.action_temperature is not None else 1.0,
        logits = action_logits,
        validate_args = False if "1.9." in torch.__version__ else None,
      ).sample()
    actions = torch.argmax(ct, dim = -1)
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
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)
      self.ckpt_path.mkdir(exist_ok = True, parents = True)

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
    self._ConfigModelParams()
    self.ckpt_step = max(0, self.loadCheckpoint())
    return

  def _ConfigModelParams(self) -> None:
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

    return

  def Train(self,
            env               : env.Environment,
            num_epochs        : int,
            num_episodes      : int, # Equivalent to batch size
            steps_per_episode : int, # Depth length of single trajectory.
            gamma             : float,
            clip              : float,
            lr                : float,
            ) -> None:
    """
    Run PPO over policy and train the agent.
    """
    # actor_optim = {
    #   'action': torch.optim.Adam(self.actor.action_parameters, lr = lr),
    #   'token' : torch.optim.Adam(self.actor.token_parameters,  lr = lr),
    # }

    # critic_optim = {
    #   'action': torch.optim.Adam(self.critic.action_parameters, lr = lr),
    #   'token' : torch.optim.Adam(self.critic.token_parameters,  lr = lr),
    # }

    # self.action_cov_var = torch.full(size = (self.qv_config.max_position_embeddings * len(interactions.ACTION_TYPE_SPACE),), fill_value = 0.5)
    # self.action_cov_mat = torch.diag(self.action_cov_var)

    # self.token_cov_var = torch.full(size = (self.tokenizer.vocab_size,), fill_value = 0.5)
    # self.token_cov_mat = torch.diag(self.token_cov_var)

    for ep in range(num_epochs):
      # Run a batch of episodes.
      action_values, action_predictions, action_policy_probs,\
      token_values, token_predictions, token_policy_probs,\
      use_lm, rewards, discounted_rewards, done = self.new_rollout(
        env, num_episodes, steps_per_episode, gamma,
      )

      input()
      # Compute Advantage at k_th iteration.
      (V_act, _, _), (V_tok, _, _) = self.evaluate_policy(batch_states, batch_actions)


      if V_act is not None:
        A_k_action = action_batch_rtgs - V_act.detach()
      else:
        A_k_action = None
      if V_tok is not None:
        A_k_token = token_batch_rtgs - V_tok.detach()
      else:
        A_k_token = None

      # Normalizing advantages isn't theoretically necessary, but in practice it decreases the variance of 
      # our advantages and makes convergence much more stable and faster. I added this because
      # solving some environments was too unstable without it.
      if A_k_action is not None:
        A_k_action = (A_k_action - A_k_action.mean()) / (A_k_action.std() + 1e-10)
        A_k_action = torch.reshape(A_k_action, (-1, 1)).repeat(1, self.qv_config.max_position_embeddings * len(interactions.ACTION_TYPE_SPACE))
      if A_k_token is not None:
        A_k_token  = (A_k_token - A_k_token.mean())   / (A_k_token.std() + 1e-10)
      
      rollout_act_probs = torch.LongTensor([[float(x) for x in a.action_logits.squeeze(0)] for a in batch_actions if a.indexed_action is not None])
      rollout_tok_probs = torch.LongTensor([[float(x) for x in a.token_logits.squeeze(0)] for a in batch_actions if a.token is not None])

      for i in range(num_updates_per_batch):
        # Calculate V_phi and pi_theta(a_t | s_t)
        (V_act, action_logits, old_act_labels), (V_tok, token_logits, old_tok_labels) = self.evaluate_policy(batch_states, batch_actions)

        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # NOTE: we just subtract the logs, which is the same as
        # dividing the values and then canceling the log with e^log.
        # For why we use log probabilities instead of actual probabilities,
        # here's a great explanation: 
        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        # TL;DR makes gradient ascent easier behind the scenes.
        if action_logits is not None:
          act_ratios = torch.exp(action_logits - rollout_act_probs)
          # Calculate surrogate losses.
          act_surr1 = act_ratios * A_k_action
          act_surr2 = torch.clamp(act_surr1, 1 - clip, 1 + clip) * A_k_action
          # Calculate actor and critic losses.
          # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
          # the performance function, but Adam minimizes the loss. So minimizing the negative
          # performance function maximizes it.
          action_loss = (-torch.min(act_surr1, act_surr2)).mean()
          act_critic_loss = torch.nn.MSELoss()(V_act, action_batch_rtgs)
          action_loss.requires_grad = True
          act_critic_loss.requires_grad = True
          print(action_loss.item())
          print(act_critic_loss.item())
          # Calculate gradients and perform backward propagation for actor network
          actor_optim['action'].zero_grad()
          action_loss.backward(retain_graph = True)
          actor_optim['action'].step()
          # Calculate gradients and perform backward propagation for critic network      
          critic_optim['action'].zero_grad()
          act_critic_loss.backward(retain_graph = True)
          critic_optim['action'].step()
        if token_logits is not None:
          tok_ratios = torch.exp(token_logits - rollout_tok_probs)
          # Calculate surrogate losses.
          tok_surr1 = tok_ratios * A_k_token
          tok_surr2 = torch.clamp(tok_surr1, 1 - clip, 1 + clip) * A_k_token
          # Calculate actor and critic losses.
          # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
          # the performance function, but Adam minimizes the loss. So minimizing the negative
          # performance function maximizes it.
          token_loss  = (-torch.min(tok_surr1, tok_surr2)).mean()
          tok_critic_loss = torch.nn.MSELoss()(V_tok, token_batch_rtgs)
          # Calculate gradients and perform backward propagation for actor network
          actor_optim['token'].zero_grad()
          token_loss.backward()
          actor_optim['token'].step()
          # Calculate gradients and perform backward propagation for critic network      
          critic_optim['token'].zero_grad()
          tok_critic_loss.backward()
          critic_optim['token'].step()

    return

  def new_rollout(self, env, num_episodes, steps_per_episode, gamma):
    """
    TODO
    """
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
    batch_feature_ids   = torch.zeros((num_episodes, steps_per_episode, feat_seq_len), dtype = torch.long)
    batch_input_ids     = torch.zeros((num_episodes, steps_per_episode, seq_len), dtype = torch.long)
    # Action, token predictions and probs, critic values.
    action_predictions  = torch.zeros((num_episodes, steps_per_episode, self.qv_config.max_position_embeddings * len(interactions.ACTION_TYPE_SPACE)), dtype = torch.long)
    action_policy_probs = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    action_values       = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    token_predictions   = torch.zeros((num_episodes, steps_per_episode, self.tokenizer.vocab_size), dtype = torch.long)
    token_policy_probs  = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    token_values        = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    use_lm              = torch.zeros((num_episodes, steps_per_episode), dtype = torch.bool)
    ## Reward placeholders.
    rewards             = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    discounted_rewards  = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    traj_disc_rewards   = torch.zeros((num_episodes, steps_per_episode), dtype = torch.float32)
    done                = torch.zeros((num_episodes, steps_per_episode), dtype = torch.bool)

    l.logger().warn("First checkpoint.")
    l.logger().warn(batch_feature_ids.shape)
    l.logger().warn(batch_input_ids.shape)
    l.logger().warn(action_predictions.shape)
    l.logger().warn(token_predictions.shape)
    l.logger().warn(token_policy_probs.shape)
    # input()
    ## Run execution loop.
    for step in range(steps_per_episode):
      ## This loop unfolds all batch_size trajectories.
      # Input tensors
      feature_ids  = batch_feature_ids[:, step]
      feature_mask = feature_ids != self.feature_tokenizer.padToken
      feature_pos  = torch.arange(feat_seq_len, dtype = torch.long).repeat(feature_ids.shape[0], 1)
      input_ids    = batch_input_ids[:, step]
      input_mask   = input_ids != self.tokenizer.padToken
      input_pos    = torch.arange(seq_len, dtype = torch.long).repeat(input_ids.shape[0], 1)

      l.logger().warn("Insert apply_normalizer here to state if step > 0")

      l.logger().warn("Second checkpoint: input to action models.")
      l.logger().warn(feature_ids.shape)
      l.logger().warn(feature_mask.shape)
      l.logger().warn(feature_pos.shape)
      l.logger().warn(input_ids.shape)
      l.logger().warn(input_mask.shape)
      l.logger().warn(input_pos.shape)
      # input()

      # Actor model returns logits of action.
      step_action_logits = self.action_actor(
        encoder_feature_ids  = feature_ids.to(pytorch.device),
        encoder_feature_mask = feature_mask.to(pytorch.device),
        encoder_position_ids = feature_pos.to(pytorch.device),
        decoder_input_ids    = input_ids.to(pytorch.device),
        decoder_input_mask   = input_mask.to(pytorch.device),
        decoder_position_ids = input_pos.to(pytorch.device),
      )['action_logits']
      # Critic model returns value logit.
      step_action_values = self.action_critic(
        encoder_feature_ids  = feature_ids.to(pytorch.device),
        encoder_feature_mask = feature_mask.to(pytorch.device),
        encoder_position_ids = feature_pos.to(pytorch.device),
        decoder_input_ids    = input_ids.to(pytorch.device),
        decoder_input_mask   = input_mask.to(pytorch.device),
        decoder_position_ids = input_pos.to(pytorch.device),
      )['action_logits']
      # Sample the most likely action.
      step_actions = self.policy.SampleActions(step_action_logits)
      # Collect the probability of said selected action, per episode.
      step_action_probs = torch.index_select(step_action_logits, -1, step_actions)

      l.logger().warn("Third checkpoint: model outputs.")
      l.logger().warn(step_action_logits.shape)
      l.logger().warn(step_action_values.shape)
      l.logger().warn(step_actions.shape)
      l.logger().warn(step_action_probs.shape)
      # input()

      ## Find which sequences need to sample a token.
      step_use_lm, masked_input_ids = env.intermediate_step(input_ids, step_actions)
      if torch.any(step_use_lm):
        ## If the language model needs to be invoked ('add' or 'replace')
        ## Fix the necessary batch of elements here.
        # Indices of starting tensors that need the LM.
        lm_indices = torch.where(step_use_lm == True)[0]

        l.logger().warn("Fourth checkpoint: LM_indices")
        l.logger().warn(lm_indices)

        # Input tensors.
        lm_feature_ids   = torch.index_select(feature_ids, 0, lm_indices)
        lm_feature_mask  = lm_feature_ids != self.feature_tokenizer.padToken
        lm_feat_pos_ids  = torch.arange(feat_seq_len, dtype = torch.long).repeat(lm_feature_ids.shape[0], 1)
        lm_input_ids     = torch.index_select(masked_input_ids, 0, lm_indices)
        lm_input_mask    = lm_input_ids != self.tokenizer.padToken
        lm_input_pos_ids = torch.arange(seq_len, dtype = torch.long).repeat(lm_input_ids.shape[0], 1)

        l.logger().warn("Fifth checkpoint: LM inputs")
        l.logger().warn(lm_feature_ids.shape)
        l.logger().warn(lm_feature_mask.shape)
        l.logger().warn(lm_feat_pos_ids.shape)
        l.logger().warn(lm_input_ids.shape)
        l.logger().warn(lm_input_mask.shape)
        l.logger().warn(lm_input_pos_ids.shape)

        # Run the token actor, get token logits.
        step_token_logits = self.token_actor(
          encoder_feature_ids  = lm_feature_ids.to(pytorch.device),
          encoder_feature_mask = lm_feature_mask.to(pytorch.device),
          encoder_position_ids = lm_feat_pos_ids.to(pytorch.device),
          decoder_input_ids    = lm_input_ids.to(pytorch.device),
          decoder_input_mask   = lm_input_mask.to(pytorch.device),
          decoder_position_ids = lm_input_pos_ids.to(pytorch.device),
        )['token_logits']
        # Keep the prediction scores only for the masked token.
        step_token_logits = torch.index_select(step_token_logits, -1, torch.where(lm_input_ids == self.tokenizer.holeToken)[0].to(pytorch.device))
        # Collect value logit from critic.
        step_token_values = self.token_critic(
          encoder_feature_ids  = lm_feature_ids.to(pytorch.device),
          encoder_feature_mask = lm_feature_mask.to(pytorch.device),
          encoder_position_ids = lm_feat_pos_ids.to(pytorch.device),
          decoder_input_ids    = lm_input_ids.to(pytorch.device),
          decoder_input_mask   = lm_input_mask.to(pytorch.device),
          decoder_position_ids = lm_input_pos_ids.to(pytorch.device),
        )['token_logits']
        # Get the critic's value only for masked index.
        step_token_values = torch.index_select(step_token_values, -1, torch.where(lm_input_ids == self.tokenizer.holeToken)[0].to(pytorch.device))
        # According to policy, select the best token.
        step_tokens = self.policy.SampleTokens(step_token_logits)
        # Get probability of said token, per episode.
        step_token_probs = torch.index_select(step_token_logits, -1, step_tokens)

        # First extend to original dimensions.
        # Store the modified - with token LM - codes to the original tensors.
        augmented_step_token_values = torch.zeros((num_episodes), dtype = torch.float32)
        augmented_step_tokens       = torch.zeros((num_episodes), dtype = torch.long)
        augmented_step_token_probs  = torch.zeros((num_episodes), dtype = torch.float32)
        for nidx, lm_idx in zip(range(step_tokens.shape[0]), lm_indices):
          augmented_step_token_values[lm_idx] = step_token_values[nidx]
          augmented_step_tokens[lm_idx]       = step_tokens[nidx]
          augmented_step_token_probs[lm_idx]  = step_token_probs[nidx]
        # Here is the appropriate storing back.
        token_values      [:, step] = augmented_step_token_values.detach().cpu()
        token_predictions [:, step] = augmented_step_tokens.detach().cpu()
        token_policy_probs[:, step] = augmented_step_token_probs.detach().cpu()

      ## Step environment and compute rewards.
      l.logger().warn("Warning, you must also step the states.")
      reward, discounted_reward, d = env.new_step(
        input_ids,
        step_actions,
        step_tokens,
        traj_disc_rewards,
        step_use_lm,
        gamma
      )

      ## Save data to rollout buffers.
      action_values      [:, step] = step_action_values.detach().cpu()
      action_predictions [:, step] = step_actions.detach().cpu()
      action_policy_probs[:, step] = step_action_probs.detach().cpu()
      use_lm             [:, step] = step_use_lm
      rewards            [:, step] = reward
      traj_disc_rewards            = discounted_reward
      discounted_rewards [:, step] = traj_disc_rewards
      done               [:, step] = d
    return (
      action_values,       # Critic action logits.
      action_predictions,  # Actor sampled label actions.
      action_policy_probs, # Actor probabilities of sampled actions.
      token_values,        # Critic token values.
      token_predictions,   # Actor sampled label tokens.
      token_policy_probs,  # Actor probabilities of sampled tokens.
      use_lm,              # Indices of actions that  required language model.
      rewards,             # Rewards of each step.
      discounted_rewards,  # Discounted rewards of each step.
      done,                # Whether this step concludes the episode.
    )

  def saveCheckpoint(self) -> None:
    """
    Save agent state.
    """
    if self.is_world_process_zero():
      ckpt_comp = lambda prefix, x: self.ckpt_path / "{}{}_model-{}.pt".format(prefix, x, self.ckpt_step)
      if self.torch_tpu_available:
        if self.pytorch.torch_xla_model.rendezvous("saving_checkpoint"):
          self.pytorch.torch_xla_model.save(self.action_actor, ckpt_comp("actor", "action"))
          self.pytorch.torch_xla_model.save(self.action_critic, ckpt_comp("critic", "action"))
        self.pytorch.torch_xla.rendezvous("saving_optimizer_states")
      else:
        if isinstance(self.action_actor, self.torch.nn.DataParallel):
          self.torch.save(self.action_actor.module.state_dict(), ckpt_comp("actor", "action"))
        else:
          self.torch.save(self.action_actor.state_dict(), ckpt_comp("action", "action"))
        if isinstance(self.action_critic, self.torch.nn.DataParallel):
          self.torch.save(self.action_critic.module.state_dict(), ckpt_comp("critic", "action"))
        else:
          self.torch.save(self.action_critic.state_dict(), ckpt_comp("critic", "action"))
        if isinstance(self.token_actor, self.torch.nn.DataParallel):
          self.torch.save(self.token_actor.module.state_dict(), ckpt_comp("actor", "token"))
        else:
          self.torch.save(self.token_actor.state_dict(), ckpt_comp("action", "token"))
        if isinstance(self.token_critic, self.torch.nn.DataParallel):
          self.torch.save(self.token_critic.module.state_dict(), ckpt_comp("critic", "token"))
        else:
          self.torch.save(self.token_critic.state_dict(), ckpt_comp("critic", "token"))

      with open(self.ckpt_path / "checkpoint.meta", 'a') as mf:
        mf.write("train_step: {}\n".format(self.ckpt_step))
    self.ckpt_step += 1
    torch.distributed.barrier()
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
        for k, v in self.torch.load(ckpt_comp("actor", "action")).items():
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
        for k, v in self.torch.load(ckpt_comp("actor", "critic")).items():
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
        for k, v in self.torch.load(ckpt_comp("token", "action")).items():
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
        for k, v in self.torch.load(ckpt_comp("token", "critic")).items():
          if k[:7] == 'module.':
            name = k[7:] # remove `module.`
          else:
            name = 'module.' + k # Add 'module.'
          new_state_dict[name] = v
        self.token_critic.load_state_dict(new_state_dict)

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
