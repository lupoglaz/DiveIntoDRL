import torch
import torch.nn as nn
import numpy as np

from Model import Actor, Critic
from Noise import Noise

class DDPG:
	def __init__(self, env):
		num_states = env.observation_space.shape[0]
		num_actions = env.action_space.shape[0]
		
		self.policy = Actor(num_states, num_actions, env.action_space).train()
		self.policy_target = Actor(num_states, num_actions, env.action_space).eval()
		self.hard_update(self.policy, self.policy_target)
		
		self.critic = Critic(num_states, num_actions).train()
		self.critic_target = Critic(num_states, num_actions).eval()
		self.hard_update(self.critic, self.critic_target)

		self.critic_loss = nn.MSELoss()

		self.batch_size = 64
		self.gamma = 0.99
		self.tau = 1e-3
		
		self.opt_critic = torch.optim.Adam(self.critic.parameters(),lr=1e-3)
		self.opt_policy = torch.optim.Adam(self.policy.parameters(),lr=1e-4)
		
	def train(self, buffer):
		b_state, b_action, b_reward, b_state_next, b_term = buffer.sample(self.batch_size)
		with torch.no_grad():
			action_target = self.policy_target(b_state_next)
			Q_prime = self.critic_target(b_state_next, action_target)

		self.opt_critic.zero_grad()
		Q = self.critic(b_state, b_action)
		L_critic = self.critic_loss(Q, b_reward + self.gamma*Q_prime*b_term)
		L_critic.backward()
		self.opt_critic.step()
		
		self.opt_policy.zero_grad()
		action = self.policy(b_state)
		L_Q = -1.0*self.critic(b_state, action).mean()
		L_Q.backward()
		self.opt_policy.step()

		self.soft_update(self.critic, self.critic_target)
		self.soft_update(self.policy, self.policy_target)

		return L_critic.item(), L_Q.item()
		
	def select_action(self, state, noise):
		with torch.no_grad():
			action = self.policy(state)
			action = noise.add_noise(action)
		return action

	def soft_update(self, src, dst):
		with torch.no_grad():
			for src_param, dst_param in zip(src.parameters(), dst.parameters()):
				dst_param.copy_(self.tau * src_param + (1.0 - self.tau) * dst_param)

	def hard_update(self, src, dst):
		with torch.no_grad():
			for src_param, dst_param in zip(src.parameters(), dst.parameters()):
				dst_param.copy_(src_param.clone())

	def load_weights(self, path):
		self.actor.load_state_dict(torch.load('{}/actor.pkl'.format(path)))
		self.critic.load_state_dict(torch.load('{}/critic.pkl'.format(path)))

	def save_model(self, path):
		torch.save(self.actor.state_dict(), '{}/actor.pkl'.format(output))
		torch.save(self.critic.state_dict(), '{}/critic.pkl'.format(output))