import numpy as np
import matplotlib.pyplot as plt
import math

def get_y(x):
	value = pow((x+2),2) - 16*math.exp(-((x-2)**2))
	return value

x = np.arange(-8,8,0.001)
y = map(lambda u: get_y(u),x)
plt.plot(x,y)

def get_grad(x):
	return (2*x+4)-16*(-2*x+4)*np.exp(-((x-2)**2))

def gradient_descent(start_x,func,grad):
	# Precision of the solution
	prec = 0.0001
	# Use a small fixed step size
	step_size = 0.01
	# max iterations
	max_iter = 100
	x_new = start_x
	res = []
	for i in xrange(max_iter):
		x_old = x_new
		# use beta = -1
		x_new = x_old - step_size*get_grad(x_new)
		f_x_new = get_y(x_new)
		f_x_old = get_y(x_old)
		res.append([x_new,f_x_new])
		if(abs(f_x_new-f_x_old) < prec):
			print "change is too small, leaving"
			return np.array(res)
	print "max number of iterations reached, leaving"
	return np.array(res)

res = gradient_descent(8,get_y,get_grad)
plt.plot(res[:,0],res[:,1],'+')
plt.show()