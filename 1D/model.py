import tensorflow as tf

class GAN_1D():

	def __init__(self, train_iterations, output_dimension):
		self.train_iterations = train_iterations
		self.output_dimension = output_dimension
		
	def MultilayerPerceptron(self, input):
		
		w1 = tf.get_variable("w0", [input.get_shape()[1], 7], initializer= tf.random_normal_initializer())
		b1 = tf.get_variable("b0", [7], initializer = tf.constant_initializer(0.0))
		w2 = tf.get_variable("w1", [7, 5], initializer = tf.random_normal_initializer())
		b2 = tf.get_variable("b1", [5], initializer = tf.constant_initializer(0.0))
		w3 = tf.get_variable("w2", [5, self.output_dimension], initializer = tf.random_normal_initializer())
		b3 = tf.get_variable("b2", [self.output_dimension], initializer = tf.constant_initializer(0.0))

		op1=tf.nn.tanh(tf.matmul(input,w1)+b1)
		op2=tf.nn.tanh(tf.matmul(op1,w2)+b2)
		op3=tf.nn.tanh(tf.matmul(op2,w3)+b3)
		
		return op3, [w1,b1,w2,b2,w3,b3]
	
	def optimizer(self, loss, varlist):
	
		batch = tf.Variable(0)
		rateoflearning = tf.train.exponential_decay(0.001,batch,self.train_iterations // 4, 0.95, staircase = True)
		optimizer = tf.train.GradientDescentOptimizer(rateoflearning).minimize(loss,global_step = batch, var_list = varlist)
		
		return optimizer