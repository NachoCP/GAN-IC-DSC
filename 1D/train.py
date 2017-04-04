import tensorflow as tf
from model import GAN_1D
from utils import *
import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

batch_size = 200
train_iterations = 20000
output_dimension = 1
mu = 0
sigma = 1

gan = GAN_1D(train_iterations, output_dimension)

# This part correspond of the pretraining phase where the discriminator
# is pretrained in order to have a more balanced decision boundary

print("First phase of the GAN_1D")
print("=====Discriminator=====")
print("Pretraining the discriminator")
with tf.variable_scope("pretrained_Discriminator"):
    input_node = tf.placeholder(tf.float32, shape=(batch_size,output_dimension))
    labels = tf.placeholder(tf.float32, shape=(batch_size, output_dimension))
    D, output = gan.MultilayerPerceptron(input_node)
    lossfunction = tf.reduce_mean(tf.square(D-labels))

pre_optimizer = gan.optimizer(lossfunction, None)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print("Initial decision boundary of the discriminator")
figure = plot_function(D,input_node, mu, sigma, batch_size, sess)
plots.title('Initial Decision Boundary')
fig = plots.figure(1)
fig.savefig('image/fig1.png')
plots.show()

loss_function_training=np.zeros(1000)
for i in range(1000):
    d=(np.random.random(batch_size)-0.5) * 10.0 
    labelsnew=norm.pdf(d,loc=mu,scale=sigma)
    loss_function_training[i],_= sess.run([lossfunction,pre_optimizer], {input_node: np.reshape(d,(batch_size,output_dimension)), 
                                    labels: np.reshape(labelsnew,(batch_size,output_dimension))})

print("Evolution of the loss function during the pretraining")
figure = plots.plot(loss_function_training, color ='black')
plots.title('Evolution of the loss function')
fig = plots.figure(1)
fig.savefig('image/fig2.png')
plots.show()

print("Final decision boundary of the discriminator after the pretraining")
plot_function(D,input_node, mu, sigma, batch_size, sess)
plots.title('Final decision boundary after the pretraining')
fig = plots.figure(1)
fig.savefig('image/fig3.png')
plots.show()

print("Storing the pretrained weight")
learned_weights = sess.run(output)

sess.close()
print("Pretraining phase finished")
print("==============================================================================")

print("Second phase of the GAN_1D")
print("=====Generator=====")
with tf.variable_scope("Generator_1D"):
    print("Initializing the variables and the normalization of the results")
    z_input_node = tf.placeholder(tf.float32, shape=(batch_size,output_dimension))
    Generator_1D,var_gen = gan.MultilayerPerceptron(z_input_node)
    Generator_1D = tf.multiply(6.0,Generator_1D)

print("=====Discriminator=====")
with tf.variable_scope("Discriminator_1D") as scope:

    print("Initializing the first discriminator with the real values")
    x_input_node = tf.placeholder(tf.float32, shape=(batch_size,output_dimension))
    op, var_dis = gan.MultilayerPerceptron(x_input_node)
    Discriminator1_1D = tf.maximum(tf.minimum(op,0.99), 0.01)
    scope.reuse_variables()
    print("Initializing the second discriminator with the values created by the Generator")
    op,var_dis = gan.MultilayerPerceptron(Generator_1D)
    Discriminator2_1D = tf.maximum(tf.minimum(op,0.99), 0.01)

print("===========")
print("Creating the different optimizers")

discriminator_objective = tf.reduce_mean(tf.log(Discriminator1_1D)+tf.log(1-Discriminator2_1D))
generator_objective = tf.reduce_mean(tf.log(Discriminator2_1D))
lossfunction_dis = 1 - discriminator_objective
lossfunction_gen = 1 - generator_objective
optimized_discriminator = gan.optimizer(lossfunction_dis, var_dis)
optimized_generator = gan.optimizer(lossfunction_gen, var_gen)

print("Loading the pretrained weights")

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
for i,v in enumerate(var_dis):
    sess.run(v.assign(learned_weights[i]))

plot_function2(Discriminator1_1D, Generator_1D, sigma, mu, batch_size, sess, x_input_node, z_input_node)
plots.title('Before Training')
fig = plots.figure(1)
fig.savefig('image/fig4.png')
plots.show()

print("Training process started")

histdis, histgen= np.zeros(train_iterations), np.zeros(train_iterations)

#Alternative between updating the generator and the discriminator
for i in range(train_iterations):
    for j in range(1):
        x= np.random.normal(mu,sigma,batch_size)
        x.sort()
        z= np.linspace(-6.0,6.0,batch_size)+np.random.random(batch_size)*0.01
        histdis[i],_=sess.run([discriminator_objective,optimized_discriminator], 
                                {x_input_node: np.reshape(x,(batch_size,output_dimension)),
                                z_input_node: np.reshape(z,(batch_size,output_dimension))})
                                
    z= np.linspace(-6.0,6.0,batch_size)+np.random.random(batch_size)*0.01
    histgen[i],_=sess.run([generator_objective,optimized_generator], {z_input_node: np.reshape(z,(batch_size,output_dimension))})
print("Training process finished")

plot_function2(Discriminator1_1D, Generator_1D, sigma, mu, batch_size, sess, x_input_node, z_input_node)
plots.title("After training")
fig = plots.figure(1)
fig.savefig('image/fig5.png')
plots.show()

