import model
import tensorflow as tf
import numpy as np
import load
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
#Information variables of the training process
sample_creation = 200 #Iteration where a sample is going to be created
show_information = 25 #Iteration where the information is going to be showed

n_epochs = 100
learning_rate = 0.0002
batch_size = 128
dim_z = 100
dim_W1 = 1024
dim_W2 = 128
dim_W3 = 64
dim_channel = 1
dim_y = 10
visualize_dimension=196


dcgan = model.GanMNIST(dim_z, dim_y, dim_W1, 
                       dim_W2, dim_W3, dim_channel, learning_rate)

print("Initialization of the model")					   
with tf.variable_scope("training_part") as scope:
    Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan.createModel(batch_size)
    session = tf.InteractiveSession()
    saver = tf.train.Saver(max_to_keep=10)

    scope.reuse_variables()
    Z_sample, Y_sample, image_sample = dcgan.sample_creator(visualize_dimension)

dis_vars = filter(lambda x: x.name.startswith(scope.name+'/dis'), tf.global_variables())
gen_vars = filter(lambda x: x.name.startswith(scope.name+'/gen'), tf.global_variables())
dis_vars = [i for i in dis_vars]
gen_vars = [i for i in gen_vars]

print("Creating the optimizers")
train_op_dis, train_op_gen = dcgan.optimizer_function(d_cost_tf, g_cost_tf, dis_vars, gen_vars, learning_rate)
tf.global_variables_initializer().run()

Z_np_sample = np.random.uniform(-1, 1, size=(visualize_dimension, dim_z))
Y_np_sample = dcgan.OneHot(np.random.randint(10, size=[visualize_dimension]))
iterations = 0
k = 2

data_loader = load.MNIST_LOADER('data/')

train_data, validation_data, test_data, train_label, validation_label, test_label = data_loader.mnist_with_valid_set()
print("==========Data Loaded=========")
print("Train set of : " + str(train_data.shape))
print("Train label of : " + str(train_label.shape))
print("Test set of : " + str(test_data.shape))
print("Test label of : " + str(test_label.shape))
print("Validation set of : " + str(validation_data.shape))
print("Validation label of : " + str(validation_label.shape))

print("Starting the training process")
for epoch in range(n_epochs):
    index = np.arange(len(train_label))
    np.random.shuffle(index)
    train_data = train_data[index]
    train_label = train_label[index]

    for start, end in zip(
            range(0, len(train_label), batch_size),
            range(batch_size, len(train_label), batch_size)
            ):

        Xs = train_data[start:end].reshape( [-1, 28, 28, 1]) / 255.
        Ys = dcgan.OneHot(train_label[start:end])
        Zs = np.random.uniform(-1, 1, size=[batch_size, dim_z]).astype(np.float32)

        if np.mod( iterations, k ) != 0:
            _, gen_loss_val = session.run([train_op_gen, g_cost_tf],feed_dict={Z_tf:Zs,Y_tf:Ys})
            discrim_loss_val, p_real_val, p_gen_val = session.run([d_cost_tf,p_real,p_gen],feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})

        else:
            _, discrim_loss_val = session.run([train_op_dis, d_cost_tf],feed_dict={Z_tf:Zs,Y_tf:Ys,image_tf:Xs})
            gen_loss_val, p_real_val, p_gen_val = session.run([g_cost_tf, p_real, p_gen],feed_dict={Z_tf:Zs, image_tf:Xs, Y_tf:Ys})

        if np.mod(iterations, show_information) == 0:
            print("========== Showing information =========")
            print("iteration:", iterations)
            print("gen loss:", gen_loss_val)
            print("discrim loss:", discrim_loss_val)
            print("Average P(real)=", p_real_val.mean())
            print("Average P(gen)=", p_gen_val.mean())

        if np.mod(iterations, sample_creation) == 0:
            generated_sample = session.run(image_sample,feed_dict={Z_sample:Z_np_sample,Y_sample:Y_np_sample})
            generated_samples = (generated_sample + 1.)/2.
            dcgan.save_visualization(generated_samples, (14,14), save_path='image/sample_%04d.jpg' % int(iterations/sample_creation))

        iterations += 1
