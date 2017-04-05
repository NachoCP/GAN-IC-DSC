import tensorflow as tf
import scipy.misc
import numpy as np

class GanMNIST():

	def __init__(self, dim_z, dim_y, 
			dim_W1, dim_W2, dim_W3, dim_channel,
			learning_rate):
			
		self.dim_z = dim_z
		self.dim_y = dim_y
		self.dim_W2 = dim_W2
		self.dim_W3 = dim_W3
		self.dim_channel = dim_channel
		self.image_shape = [28,28,1]
		
		self.GEN_kernel_W1 = [dim_z + dim_y, dim_W1]
		self.GEN_kernel_W2 = [dim_W1 + dim_y, dim_W2*7*7]
		self.GEN_kernel_W3 = [5, 5, dim_W3, dim_W2 +  dim_y]
		self.GEN_kernel_W4 = [5, 5, dim_channel, dim_W3 +  dim_y]
		
		self.DIS_kernel_W1 = [5, 5, dim_channel + dim_y, dim_W3]
		self.DIS_kernel_W2 = [5, 5, dim_W3 + dim_y, dim_W2]
		self.DIS_kernel_W3 = [dim_W2*7*7 + dim_y, dim_W1]
		self.DIS_kernel_W4 = [dim_W1 + dim_y, 1]
	
	def batchnormalization(self, X, W = None, b = None):
		eps = 1e-8
		if X.get_shape().ndims == 4:
			mean = tf.reduce_mean(X, [0,1,2])
			standar_desviation = tf.reduce_mean(tf.square(X-mean), [0,1,2])
			X = (X - mean) / tf.sqrt(standar_desviation + eps)
			
			if W is not None and b is not None:
				W = tf.reshape(W, [1,1,1,-1])
				b = tf.reshape(b, [1,1,1,-1])
				X = X*W + b
		
		elif X.get_shape().ndims == 2:
			mean = tf.reduce_mean(X, 0)
			standar_desviation = tf.reduce_mean(tf.square(X-mean), 0)
			X = (X - mean) / tf.sqrt(standar_desviation + eps)
			
			if W is not None and b is not None:
				W = tf.reshape(W, [1,-1])
				b = tf.reshape(b, [1,-1])
				X = X*W + b
		
		return X
		
	
	def leakyRelu(self, X):
		alpha = 0.2
		return tf.maximum(X,tf.multiply(X, alpha))
		
	def bce(self, x, z):
	
		x = tf.clip_by_value(x, 1e-7, 1. - 1e-7)
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = x, labels = z))
		
	# As have been said before, this function allow us to choose the correct number of ouputs we want.
	
	def MultilayerPerceptronGenerator(self, Z, Y, batch_size):
		
		gen_W1 = tf.get_variable("gen_W1", self.GEN_kernel_W1, initializer=tf.random_normal_initializer(stddev=0.02))
		gen_W2 = tf.get_variable("gen_W2", self.GEN_kernel_W2, initializer=tf.random_normal_initializer(stddev=0.02))
		gen_W3 = tf.get_variable("gen_W3", self.GEN_kernel_W3, initializer=tf.random_normal_initializer(stddev=0.02))
		gen_W4 = tf.get_variable("gen_W4", self.GEN_kernel_W4, initializer=tf.random_normal_initializer(stddev=0.02))
		
		yb = tf.reshape(Y, [batch_size, 1, 1, int(Y.get_shape()[1])])
		Z = tf.concat([Z, Y], axis=1) 
		op1 = tf.nn.relu(self.batchnormalization(tf.matmul(Z, gen_W1)))
		op1 = tf.concat([op1, Y], axis=1)
		op2 = tf.nn.relu(self.batchnormalization(tf.matmul(op1, gen_W2)))
		op2 = tf.reshape(op2, [batch_size, 7, 7, self.dim_W2])
		op2 = tf.concat([op2, yb*tf.ones([batch_size, 7, 7, int(Y.get_shape()[1])])], axis = 3)
		
		op3 = tf.nn.conv2d_transpose(op2, gen_W3, output_shape=[batch_size, 14, 14, self.dim_W3], strides=[1,2,2,1])
		op3 = tf.nn.relu(self.batchnormalization(op3))
		op3 = tf.concat([op3, yb*tf.ones([batch_size, 14, 14, Y.get_shape()[1]])], axis = 3)
		op4 = tf.nn.conv2d_transpose(op3, gen_W4, output_shape=[batch_size, 28, 28, self.dim_channel], strides=[1,2,2,1])
		
		return op4
	
	def MultilayerPerceptronDiscriminator(self, image, Y, batch_size):
		
		dis_W1 = tf.get_variable("dis_W1", self.DIS_kernel_W1, initializer=tf.random_normal_initializer(stddev=0.02))
		dis_W2 = tf.get_variable("dis_W2", self.DIS_kernel_W2, initializer=tf.random_normal_initializer(stddev=0.02))
		dis_W3 = tf.get_variable("dis_W3", self.DIS_kernel_W3, initializer=tf.random_normal_initializer(stddev=0.02))
		dis_W4 = tf.get_variable("dis_W4", self.DIS_kernel_W4, initializer=tf.random_normal_initializer(stddev=0.02))

		yb = tf.reshape(Y, tf.stack([batch_size, 1, 1, int(Y.get_shape()[1])]))
		X = tf.concat([image, yb*tf.ones([batch_size, 28, 28, int(Y.get_shape()[1])])], axis = 3)
		op1 = self.leakyRelu( tf.nn.conv2d( X, dis_W1, strides=[1, 2, 2, 1], padding='SAME'))
		op1 = tf.concat([op1, yb*tf.ones([batch_size, 14, 14, int(Y.get_shape()[1])])], axis = 3)
		op2 = self.leakyRelu( tf.nn.conv2d( op1, dis_W2, strides=[1, 2, 2, 1], padding='SAME'))
		op2 = tf.reshape(op2, [batch_size, -1])
		op2 = tf.concat([op2, Y], axis = 1)
		op3 = self.leakyRelu(self.batchnormalization(tf.matmul(op2, dis_W3)))
		op3 = tf.concat([op3, Y], axis = 1)
		
		p = tf.nn.sigmoid(tf.matmul(op3, dis_W4))
		return p, op3
	
	def sample_creator(self, dimension):
		
		Z = tf.placeholder(tf.float32, [dimension, self.dim_z])
		Y = tf.placeholder(tf.float32, [dimension, self.dim_y])

		op4 = self.MultilayerPerceptronGenerator(Z,Y,dimension)
		image = tf.nn.sigmoid(op4)
		
		return Z,Y,image
	
	def createModel(self, batch_size):
    
		Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
		Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])
		
		image_real = tf.placeholder(tf.float32, [batch_size] + self.image_shape)
		
		op4_generated = self.MultilayerPerceptronGenerator(Z,Y, batch_size)
		image_generate = tf.nn.sigmoid(op4_generated)
		
		with tf.variable_scope("discriminator_variables") as scope:
			p_real, raw_real = self.MultilayerPerceptronDiscriminator(image_real, Y, batch_size)
			scope.reuse_variables()
			p_gen, raw_gen = self.MultilayerPerceptronDiscriminator(image_generate, Y, batch_size)
		
		
		dis_cost_real = self.bce(raw_real, tf.ones_like(raw_real))
		dis_cost_gen = self.bce(raw_gen, tf.zeros_like(raw_gen))
		dis_cost = dis_cost_real + dis_cost_gen
		
		gen_cost = self.bce (raw_gen, tf.ones_like(raw_gen))
		
		return Z, Y, image_real, dis_cost, gen_cost, p_real, p_gen
	
	def OneHot(self, X, n=None, negative_class=0.):
		X = np.asarray(X).flatten()
		if n is None:
			n = np.max(X) + 1
		Xoh = np.ones((len(X), n)) * negative_class
		Xoh[np.arange(len(X)), X] = 1.
		return Xoh
    
	def save_visualization(self, X, nh_nw, save_path='tmp/sample.jpg'):
	
		h,w = X.shape[1], X.shape[2]
		img = np.zeros((h * nh_nw[0], w * nh_nw[1], 3))
		for n,x in enumerate(X):
			j = n // nh_nw[1]
			i = n % nh_nw[1]
			img[j*h:j*h+h, i*w:i*w+w, :] = x
			
			
		scipy.misc.imsave(save_path, img)
	
	def optimizer_function(self, d_cost_tf, g_cost_tf, dis_vars, gen_vars,learning_rate):
	
		train_op_dis = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=dis_vars)
		train_op_gen = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)
		
		return train_op_dis, train_op_gen