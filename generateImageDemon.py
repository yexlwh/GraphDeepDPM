from __future__ import absolute_import, division, print_function

import math

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import os
from utils import encoder, decoder
from generator import Generator
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from vdpmm_maximizeCNN import *
from dp_init import *
import math as math
# from RBF import *
from vdpmm_expectationCNNKnearset import *
from utils import discriminator
from sklearn.decomposition import PCA
from vdpmm_maximizePlusGaussian import *
from vdpmm_expectationPlusGaussian import *
from sklearn import metrics

hidden_size=1
batch_size=128

input_tensor = tf.placeholder(tf.float32, [None, 28 * 28])
xs2 = tf.placeholder(tf.float32, [None, 28 * 28])
dis=tf.placeholder(tf.float32, [1, None])
flag=tf.placeholder(tf.float32, [1, None])

with tf.variable_scope("model") as scope:
    encoded = encoder(input_tensor, hidden_size * 2)

    mean = encoded[:, :hidden_size]
    stddev = tf.sqrt(tf.square(encoded[:, hidden_size:]))

    epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])
    input_sample = mean + epsilon * stddev

    output_tensor = decoder(input_sample)

with tf.variable_scope("model") as scope:
    encoded1 = encoder(xs2, hidden_size * 2)

    mean1 = encoded1[:, :hidden_size]
    stddev1 = tf.sqrt(tf.square(encoded1[:, hidden_size:]))

    epsilon1 = tf.random_normal([tf.shape(mean1)[0], hidden_size])
    input_sample1 = mean1 + epsilon1 * stddev1

    output_tensor1 = decoder(input_sample1)

with tf.variable_scope("model", reuse=True) as scope:
    encoded = encoder(input_tensor, hidden_size * 2)

    mean = encoded[:, :hidden_size]
    stddev = tf.sqrt(tf.square(encoded[:, hidden_size:]))

    epsilon = tf.random_normal([tf.shape(mean)[0], hidden_size])
    input_sample = mean + epsilon * stddev
    sampled_tensor = decoder(input_sample)
    #sampled_tensor = decoder(tf.random_normal([batch_size*10, hidden_size]))

Nz=128;
K=2;
D=hidden_size
BPlace=tf.placeholder(tf.float32, [D, D, K])
aPlace=tf.placeholder(tf.float32, [K,1])
meanPlace=tf.placeholder(tf.float32, [K, D])
gammasPlace=tf.placeholder(tf.float32, [Nz, K])

k0=0
loss20=tf.constant(0.0)
tempU0=meanPlace[k0,:]#tf.to_float(tf.constant(uk[k,:]))
loss10=(tempU0-mean)
tempB0=BPlace[:,:,k0]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro0=tf.matmul(loss10,tempB0)
tempA0=aPlace[k0]#tf.to_float(tf.constant(params['a'][k]))
loss0=tf.reduce_mean((tf.reshape(gammasPlace[:,k0],[-1,1])*(tempA0*tempPro0*loss10+tempA0*tempB0*stddev)-0.5*tf.log(stddev))/(1e5))

k1=tf.constant(1)
tempU1=meanPlace[k1,:]#tf.to_float(tf.constant(uk[k,:]))
loss11=(tempU1-mean)
tempB1=BPlace[:,:,k1]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro1=tf.matmul(loss11,tempB1)
tempA1=aPlace[k1]#tf.to_float(tf.constant(params['a'][k]))
loss1=tf.reduce_mean((tf.reshape(gammasPlace[:,k1],[-1,1])*(tempA1*tempPro1*loss11+tempA1*tempB1*stddev)-0.5*tf.log(stddev))/(1e5))

# with tf.variable_scope("model2", reuse=True) as scope1:
#     sampled_tensor1 = decoder(tf.random_normal([batch_size, hidden_size]))
rec_loss=tf.reduce_mean(0.5 * (tf.square(mean) + tf.square(stddev) -2.0 * tf.log(stddev + 0.01) - 1.0))
vae_loss=tf.reduce_sum(-input_tensor * tf.log(output_tensor+0.01) -(1.0 - input_tensor) * tf.log(1.0 - output_tensor +0.01))
vae_loss1=tf.reduce_sum(-xs2 * tf.log(output_tensor1+0.01) -(1.0 - xs2) * tf.log(1.0 - output_tensor1 +0.01))
rec_lossH1=tf.reduce_sum(flag*tf.square(tf.reduce_sum(tf.square(mean1-mean),1)-dis))
#vae_loss=tf.reduce_mean(tf.square(input_tensor-output_tensor))
lossFinal1 = 0.1*(loss0+loss1)
loss=rec_loss+vae_loss+1*rec_lossH1+lossFinal1

#input_tensor-output_tensor
#train = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(), learning_rate=1e-4, optimizer='Adam', update_ops=[])
train = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(), learning_rate=1e-4, optimizer='Adam', update_ops=[])

init = tf.global_variables_initializer()

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
run_config.gpu_options.per_process_gpu_memory_fraction = 1/10
sess = tf.Session(config=run_config)
sess.run(init)


def generate_and_save_images( images, directory):
    '''Generates the images using the model and saves them in the directory

    Args:
        num_samples: number of samples to generate
        directory: a directory to save the images
    '''
    imgs = sess.run(sampled_tensor,{input_tensor:images})
    for k in range(imgs.shape[0]):
        imgs_folder = os.path.join(directory, 'imgs')
        if not os.path.exists(imgs_folder):
            os.makedirs(imgs_folder)

        imsave(os.path.join(imgs_folder, '%d.png') % k,
               imgs[k].reshape(28, 28))

orderedImage6 = np.load("orderedImage8.npy")
orderedImage7 = np.load("orderedImage4.npy")
images6 = orderedImage6[0:64, :]
images7 = orderedImage7[0:64, :]
images = np.concatenate((images6, images7))

pca=PCA(n_components=5,whiten=True)
newData=pca.fit_transform(images)

posMean=sess.run(mean,feed_dict={input_tensor: images})
params,gammas = vdpmm_init(posMean,K)
paramsGaussian,gammasGaussian = vdpmm_init(newData,K)
if __name__ == "__main__":


    #for i in range(100):
    #images2, _ = mnist.train.next_batch(batch_size*100)
    lambdaPos = 0.0

    for epoch in range(200):
        training_loss = 0.0

        # pbar = ProgressBar()
        # for i in range(30):
        #     images, _ = mnist.train.next_batch(batch_size)

        for i in range(1):
            # images, _ = mnist.train.next_batch(batch_size)
            flagReal = kneighbors_graph(images, 10, mode='connectivity', include_self=True)
            disReal = kneighbors_graph(images, 10, mode='distance', include_self=True)
            flagReal = flagReal.toarray()
            disReal = disReal.toarray()

            flagReal2 = (flagReal - 1)
            flagReal2 = flagReal2 * flagReal2
            disReal = 100000 * flagReal2 + flagReal * 1
            flagReal = flagReal2 + flagReal
            # sess.run(train, {input_tensor: images})
            # loss_value = sess.run(loss, {input_tensor: images})
            # training_loss = training_loss + loss_value
            for i in range(batch_size):
                Xtemp2 = np.matlib.repmat(images[i, :], batch_size, 1)
                disTemp = disReal[i, :].reshape(1, batch_size);
                flagTemp = flagReal[i, :].reshape(1, batch_size);
                sess.run(train,feed_dict={input_tensor: images, xs2: Xtemp2, dis: disTemp, flag: flagTemp, xs2: Xtemp2, BPlace: params['B'],aPlace: params['a'].reshape(K, 1), meanPlace: params['mean'], gammasPlace: gammas})
                # posGammas=sess.run(train_step2, feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp,xs: X,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})

                # sess.run(train_step,feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp})

                # sess.run(train,{input_tensor:images,xs2:Xtemp2,dis:disTemp,flag:flagTemp})
                loss_value = sess.run(loss,feed_dict={input_tensor: images, xs2: Xtemp2, dis: disTemp, flag: flagTemp, xs2: Xtemp2, BPlace: params['B'],aPlace: params['a'].reshape(K, 1), meanPlace: params['mean'], gammasPlace: gammas})
                training_loss = training_loss+loss_value
                posMean = sess.run(mean, feed_dict={input_tensor: images})
                posCov = sess.run(stddev, feed_dict={input_tensor: images})
                paramsGaussian = vdpmm_maximizePlusGaussian(newData, paramsGaussian, (lambdaPos) * gammas)
                posGaussian = vdpmm_expectationPlusGaussian(newData, paramsGaussian)
                params = vdpmm_maximizeCNN(posMean, params, (1 - lambdaPos) * gammas, posCov);
                gammas = vdpmm_expectationCNNKnearset(posMean, params, lambdaPos, (lambdaPos) * posGaussian)


        print("Loss %f" % training_loss)
        # images, _ = mnist.train.next_batch(batch_size*10)
        generate_and_save_images(images, "")
    # print(posMean);
    temp = np.max(gammas, axis=1)
    temp.shape = (Nz, 1)
    index1 = np.where(temp == gammas)
    labelsrc=index1[1]+1.0
    a = np.zeros((64, 1))
    b = np.ones((64, 1))
    labelTemp=np.concatenate((a, b))
    labelDst=labelTemp.reshape(1, 128).tolist()[0]
    print("NMI clustering accuracy",metrics.adjusted_mutual_info_score(labelDst, labelsrc))
