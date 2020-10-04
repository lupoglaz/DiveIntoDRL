import torch
import numpy as np

class Noise(object):
	def __init__(self, env_action, mu=0.0, theta=0.15, max_sigma=0.3,
				min_sigma=0.1, decay_period=5000):
		self.mu = mu
		self.theta = theta
		self.sigma = max_sigma
		self.max_sigma = max_sigma 
		self.min_sigma = min_sigma 
		self.decay_period = decay_period
		self.num_actions = env_action.shape[0]
		self.action_low = env_action.low[0]
		self.action_high = env_action.high[0]
		self.num_step = 0
		self.reset()

	def reset(self):
		self.state = np.zeros(self.num_actions)
		self.num_step = 0
	
	def state_update(self):
		x = self.state
		dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.num_actions)
	
	def add_noise(self, action):
		self.state_update()
		state = torch.from_numpy(self.state).to(dtype=action.dtype, device=action.device)
		self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma)*min(1.0, self.num_step)/self.decay_period
		self.num_step += 1
		return torch.clamp(action + state, self.action_low, self.action_high)