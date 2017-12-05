from __future__ import division,print_function
import numpy as np 
from functools import reduce

class Perceptron(object):
	"""docstring for Perceptron"""
	def __init__(self, data,labels,neurons_nums,f,iterations,rate):
		super(Perceptron, self).__init__()
		self.data = data
		self.labels = labels
		self.neurons_nums = neurons_nums
		self.f = f
		self.iterations = iterations
		self.weights = np.random.rand(self.neurons_nums)
		self.bias = np.random.rand(1)
		self.rate = rate

	def __str__(self):
		return 'weights:{}\n bias:{}'.format(self.weights,self.bias)
	def inference(self,input_data):
		return self.f(
			reduce(lambda x,y:x+y,
				map(lambda x:x[0]*x[1],zip(input_data,self.weights))) + self.bias)

	def train(self):
		train_data = zip(self.data,self.labels)
		for i in range(self.iterations):
			for x,y in train_data:
				out = self.inference(x)
				delta = y - out
				self.weights = map(
					lambda x,w:w+self.rate*delta*x,zip(x,self.weights))
				self.bias += delta * self.rate

def f(x):
	return 1 if x>0 else 0

if __name__ == "__main__":
	data = np.array([[1,1],[1,0],[0,1],[0,0]])
	labels = np.array([1,0,0,0])
	and_perceptron = Perceptron(data,labels,2,f,10,0.1)
	print(and_perceptron)
	print('1 and 2 = {}'.format(and_perceptron.inference(np.array([1,2]))))
