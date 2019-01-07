from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os 
import shutil
import time

np.random.seed(1234)

class DatasetRegression(object):
	"""docstring for ClassName"""
	def __init__(self, dim, train_size, valid_size, test_size):
		self.dim = dim
		self.w = np.random.uniform(-0.5, 0.5, (self.dim, 1))
		self.train_set, self.train_mse = self.create_data(train_size)
		self.valid_set, self.valid_mse = self.create_data(valid_size)
		self.test_set, self.test_mse = self.create_data(test_size)
		self.train_size = train_size
		self.counter = 0
  
	def create_data(self, set_size): 
		u = np.random.uniform(-5, 5, (set_size, self.dim))
		wu = np.matmul(u,self.w)
		eta = np.random.normal(loc=0, scale=2.0, size=np.shape(wu))
		v = wu + eta
		mse = (eta ** 2).mean()
		return (u,v), mse

	def next_batch(self, batch_size):
		if (self.counter + batch_size) > self.train_size:
			start = (self.counter + batch_size) % self.train_size
		else:
			start = self.counter

		self.counter = start + batch_size
		# print("self.counter", self.counter)
		# print(self.train_set.shape)
		return (self.train_set[0][start:self.counter],self.train_set[1][start:self.counter])
	
	def get_training(self):
		return self.train_set

	def get_validation(self):
		return self.valid_set

	def get_test(self):
		return self.test_set


class DatasetClassification(object):
	"""docstring for ClassName"""
	def __init__(self, d, D, train_size, valid_size, test_size):

		self.dim = d
		self.D = D

		numbers = np.random.random_integers(0, 2**self.dim - 1, size=[4]) 
		vertices = [(np.binary_repr(n, width=self.dim)) for n in numbers]
		self.vertices_array = np.array([list(v) for v in vertices]).astype(np.uint8)

		self.train_set = self.create_data(train_size)
		self.valid_set = self.create_data(valid_size)
		self.test_set = self.create_data(test_size)
		self.train_size = train_size
		self.counter = 0
  
	def create_data(self, set_size): 
		data_X = np.zeros(shape=(set_size, self.dim*self.D)).astype(np.float32)
		data_Y = np.zeros(shape=(set_size, 2)).astype(np.float32)
		for idx in range(set_size):
			ind = np.random.random_integers(0,3)
			if ind < 2:
				v = 1
			else:
				v = 0
			data_Y[idx,v] = 1.0

			u_base = self.vertices_array[ind]
			u1 = np.repeat(u_base[:,np.newaxis], int(0.05*self.D), axis=1)
			u1 = u1 + np.random.normal(loc=0, scale=1.0, size=np.shape(u1))

			u2_const = np.random.uniform(low=-1, high=1, size=(int(0.05*self.D),int(0.05*self.D)))
			u2 = np.matmul(u1, u2_const)

			u3 = np.random.normal(loc=0, scale=1.0, size=(self.dim, self.D-int(0.05*self.D)*2))
			u = np.concatenate((u1, u2, u3), axis=1)
			data_X[idx,:] = u.transpose().ravel()
		return (data_X, data_Y)

	def next_batch(self, batch_size):
		if (self.counter + batch_size) > self.train_size:
			start = (self.counter + batch_size) % self.train_size
		else:
			start = self.counter

		self.counter = start + batch_size
		return (self.train_set[0][start:self.counter],self.train_set[1][start:self.counter])
	
	def get_training(self):
		return self.train_set

	def get_validation(self):
		return self.valid_set

	def get_test(self):
		return self.test_set



def plot_loss(ax, prefix, loss1_list, loss2_list, loss_train_list, loss_valid_list, setting='classification'):

	ax.clear()

	ax.plot(range(len(loss1_list)), loss1_list, color="b", label=setting+' loss')
	ax.plot(range(len(loss2_list)), loss2_list, color="g", label='regularizer loss')
	ax.plot(range(len(loss_train_list)), loss_train_list, color="r", label='joint loss')
	ax.plot(range(len(loss_valid_list)), loss_valid_list, color="k", label='validation loss')
	ax.grid(True)
	if setting=='classification':
		plt.ylim([-0.0,2.0])
	else:
		plt.ylim([-5.0,5.0])
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.legend()

	plt.draw()
	plt.pause(1e-4)
	if len(prefix):
		plt.savefig(prefix + '.png')
	# plt.show()
