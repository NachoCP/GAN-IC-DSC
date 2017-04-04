import matplotlib.pyplot as plots
import numpy as np
from scipy.stats import norm

def plot_function(D,input_node, mu, sigma, batch_size, sess):
    figure,axis=plots.subplots(1)

    xaxis=np.linspace(-6,6,1000)
    axis.plot(xaxis, norm.pdf(xaxis,loc=mu,scale=sigma), label='p_distribution')

    r=1000
    xaxis=np.linspace(-6,6,r)

    decisor_beginning=np.zeros((r,1)) 
    for i in range(int(r/batch_size)):
        x=np.reshape(xaxis[batch_size*i:batch_size*(i+1)],(batch_size,1))
        decisor_beginning[batch_size*i:batch_size*(i+1)]=sess.run(D,{input_node: x})

    axis.plot(xaxis, decisor_beginning, label='decision boundary', color ='black')
    axis.set_ylim(0,1.1)
    plots.legend()


def plot_function2(dis, gen, sigma, mu, batch_size, sess, x_input_node, z_input_node):

    figure,axis=plots.subplots(1)

    xaxis=np.linspace(-6,6,1000)
    axis.plot(xaxis, norm.pdf(xaxis,loc=mu,scale=sigma), label='p_original_distribution')

    r=1000 
    xaxis=np.linspace(-6,6,r)
    decisor=np.zeros((r,1))

    for i in range(int(r/batch_size)):
        x=np.reshape(xaxis[batch_size*i:batch_size*(i+1)],(batch_size,1))
        decisor[batch_size*i:batch_size*(i+1)]=sess.run(dis,{x_input_node: x})

    axis.plot(xaxis, decisor, label='decision boundary',color='black')

    zs=np.linspace(-6,6,r)
    generated=np.zeros((r,1)) 
    for i in range(int(r/batch_size)):
        z=np.reshape(zs[batch_size*i:batch_size*(i+1)],(batch_size,1))
        generated[batch_size*i:batch_size*(i+1)]=sess.run(gen,{z_input_node: z})
    histo_gendata, binedges = np.histogram(generated)
    axis.plot(np.linspace(-6,6,10), histo_gendata/float(r), label='data_generated',color ='yellow')

    axis.set_ylim(0,1.1)
    plots.legend
