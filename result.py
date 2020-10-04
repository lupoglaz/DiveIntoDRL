import sys
import numpy as np
import random
import seaborn as sea
sea.set_style("whitegrid")
import matplotlib.pylab as plt
from celluloid import Camera

from Logger import Logger

def plot_hist(np_hist, label):
	hist, bins = np_hist
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	p = plt.bar(center, hist, align='center', width=width, color='red', edgecolor='black')
	plt.legend(p, [label])

if __name__=='__main__':
	log = Logger('Logs/debug', create=False)
	f = plt.figure(figsize=(12,6))
	camera = Camera(f)
	lc = []
	lq = []
	x= []
	av_rw = []
	min_rw = []
	max_rw = []
	for n,dict in enumerate(log):
		x.append(n)
		av_rw.append(dict['av_rw'])
		min_rw.append(dict['min_rw'])
		max_rw.append(dict['max_rw'])
		if dict['lc'] != None:
			lq.append(dict['lq'])
			lc.append(dict['lc'])

		plt.subplot(3, 2, 1)
		plot_hist(dict['ah'], 'Buffer actions')
		plt.subplot(3, 2, 2)
		plot_hist(dict['rh'], 'Buffer rewards')
		plt.subplot(3, 2, 3)
		plot_hist(dict['trh'], 'Buffer state transitions')
		plt.subplot(3, 2, 4)
		p = plt.plot(lq, color='red')
		plt.legend(p, ['Qloss'])
		plt.subplot(3, 2, 5)
		p = plt.plot(lc, color='red')
		plt.legend(p, ['Policy loss'])
		plt.subplot(3, 2, 6)
		p = plt.plot(av_rw, color='red')
		plt.fill_between(x, min_rw, max_rw, color='gray', alpha=0.2)
		plt.legend(p, ['Average episode reward'])
		# plt.show()
		# sys.exit()
		camera.snap()
		
	animation = camera.animate()
	plt.show()
	# animation.show()
	# animation.save('animation.mp4')