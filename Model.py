import torch
import torch.nn as nn

def weights_init(m):
	if isinstance(m, nn.Linear):
		torch.nn.init.xavier_uniform_(m.weight.data)
		torch.nn.init.uniform_(m.bias.data)

class Actor(nn.Module):
	def __init__(self, num_states, num_actions, env_action, hidden=[400, 300]):
		super(Actor, self).__init__()
		self.env_action = env_action
		self.net = nn.Sequential(
			nn.Linear(num_states, hidden[0]),
			nn.ReLU(),
			nn.Linear(hidden[0], hidden[1]),
			nn.ReLU(),
			nn.Linear(hidden[1], num_actions),
			nn.Tanh()
		)
		self.apply(weights_init)

	def forward(self, x):
		span = self.env_action.high[0]-self.env_action.low[0]
		return span*self.net(x)

class Critic(nn.Module):
	def __init__(self, num_states, num_actions, hidden=[400, 300]):
		super(Critic, self).__init__()
		self.state_net = nn.Sequential(
			nn.Linear(num_states, hidden[0]),
			nn.ReLU()
		)
		self.state_action_net = nn.Sequential(
			nn.Linear(hidden[0] + num_actions, hidden[1]),
			nn.ReLU(),
			nn.Linear(hidden[1], 1)
		)
		self.apply(weights_init)
		
	def forward(self, state, action):
		state_prime = self.state_net(state)
		state_action = torch.cat([state_prime, action], dim=-1)
		return self.state_action_net(state_action).squeeze()