import numpy as np

d = 5
P = 1
D = 100


numbers = np.random.random_integers(0, 2**d - 1, size=[4]) 
vertices = [(np.binary_repr(n, width=d)) for n in numbers]
vertices_array = np.array([list(v) for v in vertices]).astype(np.uint8)


data_X = np.zeros(shape=(P, d*D))
data_Y = np.zeros(shape=(P, 1))

for p in range(P):
	ind = np.random.random_integers(0,3)
	# print(ind)

	if ind < 2:
		v = 1
	else:
		v = 0
	# print(v)
	data_Y[p,0] = v

	u_base = vertices_array[ind]
	# print(u_base)
	u1 = np.repeat(u_base[:,np.newaxis], int(0.05*D), axis=1)
	u1 = u1 + np.random.normal(loc=0, scale=1.0, size=np.shape(u1))
	# print(u1)

	u2_const = np.random.uniform(low=-1, high=1, size=(int(0.05*D),int(0.05*D)))
	# print(u2_const)
	u2 = np.matmul(u1, u2_const)
	# print(u2)

	u3 = np.random.normal(loc=0, scale=1.0, size=(d, int(0.9*D)))
	u = np.concatenate((u1, u2, u3), axis=1)
	data_X[p,:] = u.transpose().ravel()
	# print("U")
	# print(u.shape)

