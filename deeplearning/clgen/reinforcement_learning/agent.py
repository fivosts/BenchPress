"""
Agents module for reinforcement learning.
"""
import pathlib
import typing
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
  def __init__(self, action_temp: float, idx_temp: float, token_temp: float):
    self.action_temperature = action_temp
    self.index_temperature  = idx_temp
    self.token_temperature  = token_temp
    return

  def SelectAction(self,
                   type_logits        : torch.FloatTensor,
                   index_logits       : torch.Tensor,
                   action_temperature : float,
                   index_temperature  : float,
                   ) -> typing.Tuple[int, int]:
    """
    Get the Q-Values for action and apply policy on it.
    """
    return 0, 0
    raise NotImplementedError
    return action_type

  def SelectToken(self,
                  token_logits : torch.FloatTensor,
                  temperature  : float,
                  ) -> int:
    """
    Get logit predictions for token and apply policy on it.
    """
    ct = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
        temperature = temperature if temperature is not None else 1.0,
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
    )
    self.q_model = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path, is_critic = False,
    )
    self.critic_model = model.QValuesModel(
      self.language_model, self.feature_tokenizer, self.qv_config, self.cache_path, is_critic = True
    )
    self.policy  = Policy(
      action_temp = self.config.agent.action_qv.action_type_temperature_micros / 10e6,
      idx_temp    = self.config.agent.action_qv.action_index_temperature_micros / 10e6,
      token_temp  = self.config.agent.action_lm.token_temperature_micros / 10e6,
    )
    self.loadCheckpoint()
    return

  def Train(self,
            env                   : env.Environment,
            num_epochs            : int,
            num_updates_per_batch : int,
            timesteps_per_batch   : int,
            max_timesteps_per_episode : int,
            ) -> None:
    """
    Run PPO over policy and train the agent.
    """
    for ep in range(num_epochs):
      # Run a batch of episodes.
      batch_states, batch_actions, batch_logits, batch_rtgs, batch_lens = self.rollout(
        env, timesteps_per_batch, max_timesteps_per_episode
      )

      # Compute Advantage at k_th iteration.
      Val, _ = self.evaluate_policy(batch_states, batch_actions)
      A_k = batch_rtgs - Val.detach()

			# Normalizing advantages isn't theoretically necessary, but in practice it decreases the variance of 
			# our advantages and makes convergence much more stable and faster. I added this because
			# solving some environments was too unstable without it.
      A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

      for i in range(num_updates_per_batch):
				# Calculate V_phi and pi_theta(a_t | s_t)
        Val, log_probs = self.evaluate(batch_states, batch_actions)

        # Calculate the ratio pi_theta(a_t | s_t) / pi_theta_k(a_t | s_t)
        # NOTE: we just subtract the logs, which is the same as
        # dividing the values and then canceling the log with e^log.
        # For why we use log probabilities instead of actual probabilities,
        # here's a great explanation: 
        # https://cs.stackexchange.com/questions/70518/why-do-we-use-the-log-in-gradient-based-reinforcement-algorithms
        # TL;DR makes gradient ascent easier behind the scenes.
        ratios = torch.exp(log_probs - batch_logits)

        # Calculate surrogate losses.
        surr1 = ratios * A_k
        surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

        # Calculate actor and critic losses.
        # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
        # the performance function, but Adam minimizes the loss. So minimizing the negative
        # performance function maximizes it.
        actor_loss = (-torch.min(surr1, surr2)).mean()
        critic_loss = torch.nn.MSELoss()(Val, batch_rtgs)

        raise NotImplementedError
        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
    return

  def rollout(self,
              env: env.Environment,
              timesteps_per_batch : int,
              max_timesteps_per_episode : int,
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
    batch_states      = []
    batch_act_types   = []
    batch_type_logs   = []
    batch_act_idxs    = []
    batch_idxs_logs   = []
    batch_act_tokens  = []
    batch_tokens_logs = []
    batch_rews        = []
    batch_rtgs        = []
    batch_lens        = []

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
        batch_acts.append(action)
        batch_log_probs.append(log_prob)

        # If the environment tells us the episode is terminated, break
        if done:
          break

      # Track episodic lengths and rewards
      batch_lens.append(ep_t + 1)
      batch_rews.append(ep_rews)

    # Reshape data as tensors in the shape specified in function description, before returning
    batch_states = torch.tensor(batch_states, dtype=torch.float)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    batch_rtgs = self.compute_rtgs(batch_rews)                                                              # ALG STEP 4

    # Log the episodic returns and episodic lengths in this batch.
    self.logger['batch_rews'] = batch_rews
    self.logger['batch_lens'] = batch_lens

    return batch_states, batch_acts, batch_log_probs, batch_rtgs, batch_lens

  def evaluate_policy(self, states, actions) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
    V = self.critic(states).squeeze()

    # Calculate the log probabilities of batch actions using most recent actor network.
    # This segment of code is similar to that in get_action()
    mean = self.actor(states)
    dist = torch.distributions.MultivariateNormal(mean, self.cov_mat)
    log_probs = dist.log_prob(actions)

    # Return the value vector V of each observation in the batch
    # and log probabilities log_probs of each action in the batch
    return V, log_probs

  def make_action(self, state: interactions.State) -> interactions.Action:
    """
    Agent collects the current state by the environment
    and picks the right action.
    """
    logits = self.q_model.SampleAction(state)
    action_logits = logits['action_logits'].cpu().numpy()
    index_logits  = logits['index_logits'].cpu().numpy()
    action_type, action_index  = self.policy.SelectAction(
      action_logits, index_logits,
      self.qv_config.action_type_temperature,
      self.qv_config.action_index_temperature,
    )

    comment = "Action: {}".format(interactions.ACTION_TYPE_MAP[action_type])

    if action_type == interactions.ACTION_TYPE_SPACE['ADD']:
      logits = self.q_model.SampleToken(
        state, action_index, self.tokenizer, self.feature_tokenizer
      )
      token_logits = logits['prediction_logits'][:,action_index]
      token        = self.policy.SelectToken(token_logits, self.qv_config.token_temperature).cpu().numpy()
      token_logits = token_logits.cpu().numpy()
      comment      += ", index: {}, token: '{}'".format(action_index, self.tokenizer.decoder[int(token)])
    elif action_type == interactions.ACTION_TYPE_SPACE['REM']:
      token_logits, token = None, None
      comment += ", index: {}".format(action_index)
    elif action_type == interactions.ACTION_TYPE_SPACE['COMP']:
      token_logits, token = None, None
    else:
      raise ValueError("Invalid action_type: {}".format(action_type))

    return interactions.Action(
      action_type         = action_type,
      action_type_logits  = action_logits,
      action_index        = action_index,
      action_index_logits = index_logits,
      token_type          = token,
      token_type_logits   = token_logits,
      comment             = comment,
    )

  def compute_rtgs(self, batch_rews):
    """
    Compute the Reward-To-Go of each timestep in a batch given the rewards.

    Parameters:
      batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

    Return:
      batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    """
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    batch_rtgs = []

    # Iterate through each episode
    for ep_rews in reversed(batch_rews):

      discounted_reward = 0 # The discounted reward so far

      # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
      # discounted return (think about why it would be harder starting from the beginning)
      for rew in reversed(ep_rews):
        discounted_reward = rew + discounted_reward * self.gamma
        batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    return batch_rtgs

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
    return
  
  def loadCheckpoint(self) -> None:
    """
    Load agent state.
    """
    return
