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

  def SelectAction(self, action_logits: torch.FloatTensor) -> typing.Tuple[int, int]:
    """
    Get the Q-Values for action and apply policy on it.
    """
    ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = self.action_temperature if self.action_temperature is not None else 1.0,
        logits = action_logits,
        validate_args = False if "1.9." in torch.__version__ else None,
      ).sample()
    likely = torch.argmax(ct, dim = -1)
    action, index = likely % len(interactions.ACTION_TYPE_SPACE), likely // len(interactions.ACTION_TYPE_SPACE)
    return int(action), int(index), int(likely)

  def SelectToken(self, token_logits: torch.FloatTensor) -> int:
    """
    Get logit predictions for token and apply policy on it.
    """
    ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = self.token_temperature if self.token_temperature is not None else 1.0,
        logits = token_logits,
        validate_args = False if "1.9." in torch.__version__ else None,
      ).sample()
    return torch.argmax(ct, dim = -1)

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
    if environment.WORLD_RANK == 0:
      self.cache_path.mkdir(exist_ok = True, parents = True)

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
    self.actor = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path, is_critic = False,
    )
    self.critic = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path, is_critic = True
    )
    self.policy  = Policy(
      action_temp = self.qv_config.action_temperature,
      token_temp  = self.qv_config.token_temperature,
    )
    self.loadCheckpoint()
    return

  def Train(self,
            env                       : env.Environment,
            num_epochs                : int,
            num_updates_per_batch     : int,
            timesteps_per_batch       : int,
            max_timesteps_per_episode : int,
            gamma                     : float,
            clip                      : float,
            lr                        : float,
            ) -> None:
    """
    Run PPO over policy and train the agent.
    """
    actor_optim = {
      'action': torch.optim.Adam(self.actor.action_parameters, lr = lr),
      'token' : torch.optim.Adam(self.actor.token_parameters,  lr = lr),
    }

    critic_optim = {
      'action': torch.optim.Adam(self.critic.action_parameters, lr = lr),
      'token' : torch.optim.Adam(self.critic.token_parameters,  lr = lr),
    }

    self.action_cov_var = torch.full(size = (self.qv_config.max_position_embeddings * len(interactions.ACTION_TYPE_SPACE),), fill_value = 0.5)
    self.action_cov_mat = torch.diag(self.action_cov_var)

    self.token_cov_var = torch.full(size = (self.tokenizer.vocab_size,), fill_value = 0.5)
    self.token_cov_mat = torch.diag(self.token_cov_var)

    for ep in range(num_epochs):
      # Run a batch of episodes.
      batch_states, batch_actions, action_batch_rtgs, token_batch_rtgs, batch_lens = self.rollout(
        env, timesteps_per_batch, max_timesteps_per_episode, gamma,
      )
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

      print("Back from eval policy.")
      print(A_k_action)
      print(A_k_token)

			# Normalizing advantages isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
      if A_k_action is not None:
        A_k_action = (A_k_action - A_k_action.mean()) / (A_k_action.std() + 1e-10)
      if A_k_token is not None:
        A_k_token  = (A_k_token - A_k_token.mean())   / (A_k_token.std() + 1e-10)

      rollout_act_labels = torch.LongTensor([a.indexed_action for a in batch_actions if a.indexed_action is not None])
      rollout_tok_labels = torch.LongTensor([a.token for a in batch_actions if a.token is not None])

      for i in range(num_updates_per_batch):
				# Calculate V_phi and pi_theta(a_t | s_t)
        (V_act, action_labels, old_act_labels), (V_tok, token_labels, old_tok_labels) = self.evaluate_policy(batch_states, batch_actions)

        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # NOTE: we just subtract the logs, which is the same as
        # dividing the values and then canceling the log with e^log.
        # For why we use log probabilities instead of actual probabilities,
        # here's a great explanation: 
        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        # TL;DR makes gradient ascent easier behind the scenes.
        print("###########")
        print(action_labels.shape)
        print(rollout_act_labels.shape)
        print("###########")
        input()
        if action_labels is not None:
          act_ratios = torch.exp(action_labels - rollout_act_labels)
          # Calculate surrogate losses.
          act_surr1 = act_ratios * A_k_action
          act_surr2 = torch.clamp(act_surr1, 1 - clip, 1 + clip) * A_k_action
          # Calculate actor and critic losses.
          # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
          # the performance function, but Adam minimizes the loss. So minimizing the negative
          # performance function maximizes it.
          action_loss = (-torch.min(act_surr1, act_surr2)).mean()
          act_critic_loss = torch.nn.MSELoss()(V_act, action_batch_rtgs)
          # Calculate gradients and perform backward propagation for actor network
          actor_optim['action'].zero_grad()
          action_loss.backward(retain_graph = True)
          actor_optim['action'].step()
          # Calculate gradients and perform backward propagation for critic network      
          critic_optim['action'].zero_grad()
          act_critic_loss.backward(retain_graph = True)
          critic_optim['action'].step()
        if token_labels is not None:
          tok_ratios = torch.exp(token_labels - rollout_tok_labels)
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

  def rollout(self,
              env                       : env.Environment,
              timesteps_per_batch       : int,
              max_timesteps_per_episode : int,
              gamma                     : float,
              ) -> typing.Tuple:
    """
    Too many transformers references, I'm sorry. This is where we collect the batch of data
    from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
    of data each time we iterate the actor/critic networks.

    Parameters:
      None

    Return:
      batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
      batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
      batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
      batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
      batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
    """
    # Batch data. For more details, check function header.
    batch_states  = []
    batch_actions = []
    batch_rews    = []
    batch_lens    = []

    # Episodic data. Keeps track of rewards per episode, will get cleared
    # upon each new episode
    ep_rews = []

    t = 0 # Keeps track of how many timesteps we've run so far this batch

    # Keep simulating until we've run more than or equal to specified timesteps per batch
    while t < timesteps_per_batch:
      ep_rews = [] # rewards collected per episode

      # Reset the environment. sNote that obs is short for observation.
      state = env.reset()
      done = False

      # Run an episode for a maximum of max_timesteps_per_episode timesteps
      for ep_t in range(max_timesteps_per_episode):
        t += 1 # Increment timesteps ran this batch so far
        # Track observations in this batch
        batch_states.append(state)

        # Calculate action and make a step in the env. 
        # Note that rew is short for reward.
        action = self.make_action(state)
        state, rew, done, _ = env.step(action)

        # Track recent reward, action, and action log probability
        ep_rews.append(rew)
        batch_actions.append(action)

        # If the environment tells us the episode is terminated, break
        if done:
          break

      # Track episodic lengths and rewards
      batch_lens.append(ep_t + 1)
      batch_rews.append(ep_rews)

    action_batch_rtgs, token_batch_rtgs = self.compute_rtgs(batch_rews, gamma) # ALG STEP 4
    return batch_states, batch_actions, action_batch_rtgs, token_batch_rtgs, batch_lens

  def evaluate_policy(self,
                      states  : typing.List[interactions.State], 
                      actions : typing.List[interactions.Action]
                      ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate the values of each observation, and the log probs of
    each action in the most recent batch with the most recent
    iteration of the actor network. Should be called from learn.

    Parameters:
      batch_obs - the observations from the most recently collected batch as a tensor.
            Shape: (number of timesteps in batch, dimension of observation)
      batch_acts - the actions from the most recently collected batch as a tensor.
            Shape: (number of timesteps in batch, dimension of action)

    Return:
      V - the predicted values of batch_obs
      log_probs - the log probabilities of the actions taken in batch_acts given batch_obs
    """
    # Query critic network for a value V for each batch_obs. Shape of V should be same as batch_rtgs
    V_actions, V_tokens = [], []
    old_action_labels, old_token_labels = [], []
    action_labels, token_labels = [], []

    for state, action in tqdm.tqdm(zip(states, actions), total = len(states), desc = "Evaluate Policy"):

      ## Re-collect the action from the updated model.
      updated_action = self.make_action(state)
      critic_logits  = self.critic.SampleAction(state)

      ## Store a) critic values, b) updated batch action log probs, c) old log probs to batch.
      V_actions        .append(critic_logits['action_logits'].cpu())
      action_labels    .append(updated_action.indexed_action)
      old_action_labels.append(action.indexed_action)

      ## If there was any meaningful token addition.
      if updated_action.token_logits is not None:
        critic_logits = self.critic.SampleToken(
          state, updated_action.index, self.tokenizer, self.feature_tokenizer,
        )
        V_tokens        .append(critic_logits['token_logits'][:,updated_action.index].cpu())
        token_labels    .append(updated_action.token)
        old_token_labels.append(action.token)

    if len(old_action_labels) > 0:
      V_actions            = torch.FloatTensor(V_actions)
      action_labels     = torch.LongTensor(action_labels)
      old_action_labels = torch.LongTensor(old_action_labels)
    else:
      V_actions         = None
      action_labels     = None
      old_action_labels = None
    
    if len(old_token_labels) > 0:
      token_labels = torch.FloatTensor(token_labels)
      old_token_labels     = torch.LongTensor(old_token_labels)
    else:
      V_tokens         = None
      token_labels     = None
      old_token_labels = None
    # Return the value vector V of each observation in the batch
    # and log probabilities log_probs of each action in the batch
    return (V_actions, action_labels, old_action_labels), (V_tokens, token_labels, old_token_labels)

  def make_action(self, state: interactions.State) -> interactions.Action:
    """
    Agent collects the current state by the environment
    and picks the right action.
    """
    output = self.actor.SampleAction(state)
    action_logits = output['action_logits'].cpu()
    action_type, action_index, indexed_action = self.policy.SelectAction(action_logits)
    action_logits = action_logits.numpy()
    comment = "Action: {}".format(interactions.ACTION_TYPE_MAP[action_type])

    if action_type == interactions.ACTION_TYPE_SPACE['ADD']:
      actual_length = np.where(np.array(state.encoded_code) == self.tokenizer.endToken)[0][0]
      if action_index < actual_length:
        logits = self.actor.SampleToken(
          state, action_index, self.tokenizer, self.feature_tokenizer
        )
        token_logits = logits['token_logits'][:,action_index]
        token        = self.policy.SelectToken(token_logits).cpu().numpy()
        token_logits = token_logits.cpu().numpy()
        comment      += ", index: {}, token: '{}'".format(action_index, self.tokenizer.decoder[int(token)])
      else:
        token_logits, token = None, None
        comment += ", index: {} - out of bounds".format(action_index)
    elif action_type == interactions.ACTION_TYPE_SPACE['REM']:
      token_logits, token = None, None
      comment += ", index: {}".format(action_index)
    elif action_type == interactions.ACTION_TYPE_SPACE['COMP']:
      token_logits, token = None, None
    elif action_type == interactions.ACTION_TYPE_SPACE['REPLACE']:
      actual_length = np.where(np.array(state.encoded_code) == self.tokenizer.endToken)[0][0]
      if action_index < actual_length:
        logits = self.actor.SampleToken(
          state,
          action_index,
          self.tokenizer,
          self.feature_tokenizer,
          replace_token = True,
        )
        token_logits = logits['token_logits'][:,action_index]
        token        = self.policy.SelectToken(token_logits).cpu().numpy()
        token_logits = token_logits.cpu().numpy()
        comment += ", index: {}, token: '{}' -> '{}'".format(action_index, self.tokenizer.decoder[int(state.encoded_code[action_index])], self.tokenizer.decoder[int(token)])
      else:
        token_logits, token = None, None
        comment += ", index {} - out of bounds".format(action_index)
    else:
      raise ValueError("Invalid action_type: {}".format(action_type))

    return interactions.Action(
      action         = action_type,
      index          = action_index,
      indexed_action = indexed_action,
      action_logits  = action_logits,
      token          = token,
      token_logits   = token_logits,
      comment        = comment,
    )

  def compute_rtgs(self, batch_rews, gamma):
    """
    Compute the Reward-To-Go of each timestep in a batch given the rewards.

    Parameters:
      batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

    Return:
      batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    """
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    action_batch_rtgs = []
    token_batch_rtgs  = []

    # Iterate through each episode
    for ep_rews in reversed(batch_rews):

      action_discounted_reward = 0 # The discounted reward for actions.
      token_discounted_reward  = 0 # The discounted reward for token additions.

      # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
      # discounted return (think about why it would be harder starting from the beginning)
      for rew in reversed(ep_rews):
        action_discounted_reward = rew.value + action_discounted_reward * gamma
        action_batch_rtgs.insert(0, action_discounted_reward)
        if rew.action.token_logits is not None:
          token_discounted_reward = rew.value + token_discounted_reward * gamma
          token_batch_rtgs.insert(0, token_discounted_reward)
    # Convert the rewards-to-go into a tensor
    action_batch_rtgs = torch.tensor(action_batch_rtgs, dtype=torch.float)
    token_batch_rtgs = torch.tensor(token_batch_rtgs, dtype=torch.float)
    return action_batch_rtgs, token_batch_rtgs

  def update_agent(self, input_ids: typing.Dict[str, torch.Tensor]) -> None:
    """
    Train the agent on the new episodes.
    """
    self.q_model.Train(input_ids)
    return

  def saveCheckpoint(self) -> None:
    """
    Save agent state.
    """
    self.actor.saveCheckpoint(prefix = "actor")
    self.critic.saveCheckpoint(prefix = "critic")
    return
  
  def loadCheckpoint(self) -> None:
    """
    Load agent state.
    """
    self.actor.loadCheckpoint(prefix = "actor")
    self.critic.loadCheckpoint(prefix = "critic")
    return
