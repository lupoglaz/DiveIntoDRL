import gym
import numpy as np
import random
import matplotlib.pylab as plt

import torch
import torch.nn as nn
from DDGP import DDPG
from ReplayBuffer import ReplayBuffer
from Noise import Noise
from Logger import Logger

if __name__ == '__main__':
	num_warmup = 5000
	num_train = 30000
	num_eval = 5000
	buffer_length = 30000

	env = gym.make('Pendulum-v0')
	# env = gym.make('MountainCarContinuous-v0')
	ddpg = DDPG(env)
	logger = Logger('Logs/debug')
	buffer = ReplayBuffer(buffer_length)
	state = env.reset()
	state = torch.from_numpy(state).to(dtype=torch.float32)

	noise = Noise(env.action_space)
	noise.reset()

	loss_critic = []
	loss_q = []
	rewards = []
	
	episode = 0
	for training_step in range(num_train + buffer.max_length + num_eval):
		
		action = ddpg.select_action(state, noise)
		state_next, reward, term, _ = env.step(action.cpu().numpy())
		state_next = torch.from_numpy(state_next).to(dtype=state.dtype, device=state.device)
		buffer.append(state, action, reward, state_next, term)

		if training_step>num_warmup and training_step<=(num_train+num_warmup):
			critic_loss, policy_loss = ddpg.train(buffer)
			loss_critic.append(critic_loss)
			loss_q.append(policy_loss)
		
		elif training_step>(num_train+num_warmup):
			env.render()

		state = state_next
		if not reward is None:
			rewards.append(reward)

		if term:
			state = env.reset()
			state = torch.from_numpy(state).to(dtype=torch.float32)
			noise.reset()
			print(f'Step: {training_step} / Episode: {episode} / Score: {np.sum(rewards)}')
			
			
			ah, rh, trh = buffer.get_histograms()
			if len(loss_critic) > 0:
				lc = np.average(loss_critic)
				lq = np.average(loss_q)
			else:
				lc = None
				lq = None
			av_rw = np.average(rewards)
			min_rw = np.min(rewards)
			max_rw = np.max(rewards)
			
			logger.log({"ah":ah, "rh":rh, "trh":trh, "lc":lc, "lq":lq, "av_rw":av_rw, "min_rw":min_rw, "max_rw":max_rw}, episode)

			loss_critic = []
			loss_q = []
			rewards = []
			episode += 1

	env.close()