from pathlib import Path
import _pickle as pkl 

class Logger:
	def __init__(self, dir, create=True):
		self.dir = Path(dir)
		self.num = 0
		if not self.dir.exists() and create:
			self.dir.mkdir(parents=True)
	
	def log(self, dict, num):
		log_file = self.dir / f'{num}.pkl'
		with log_file.open('bw') as fout:
			pkl.dump(dict, fout)
	
	def __iter__(self):
		self.num = 0
		return self
	
	def __next__(self):
		log_file = self.dir / f'{self.num}.pkl'
		self.num+=1
		if not log_file.exists():
			raise StopIteration()
		with log_file.open('br') as fin:
			dict = pkl.load(fin)
		return dict
		