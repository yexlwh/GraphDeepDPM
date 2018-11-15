# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:58:24 2017

@author: yexlwh
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt;
import sklearn as skl
# from dataCombinedX1 import dataCombineX
from mpl_toolkits.mplot3d import Axes3D
from vdpmm_maximizePlusGaussian import *
from vdpmm_expectationPlusGaussian import *
from sklearn.neighbors import kneighbors_graph

# from vdpmm_expectationCNN import *
from vdpmm_maximizeCNN import *
from dp_init import *
import math as math
# from RBF import *
from vdpmm_expectationCNNKnearset import *
from sklearn import cluster, datasets
from sklearn import metrics

def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.get_variable("weights",[in_size, out_size])
    biases = tf.get_variable("biases",[1, out_size])
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
def encoderVAE(inputs, hiden_size):
    with tf.variable_scope("foo"):
        l1 = add_layer(inputs, 2, 3, activation_function=tf.nn.sigmoid)
    # with tf.variable_scope("hfoo"):
    #     hl1 = add_layer(h1, 7, 3, activation_function=tf.nn.relu)
    # with tf.variable_scope("hfoo1"):
    #     l1 = add_layer(hl1, 3, 3, activation_function=tf.nn.relu)
    with tf.variable_scope("foo1"):
        l2 = add_layer(l1, 3, 4, activation_function=tf.nn.relu)
    with tf.variable_scope("foo3"):
        l3 = add_layer(l2, 4, 2, activation_function=tf.nn.relu)
    with tf.variable_scope("foo4"):
        prediction = add_layer(l3,2, hiden_size*2)#,activation_function=tf.nn.sigmoid)
    return prediction

def decoderVAE(inputs, hiden_size):
    with tf.variable_scope("foo5"):
        l1 = add_layer(inputs, hiden_size, 2, activation_function=tf.nn.sigmoid)
    with tf.variable_scope("foo6"):
        l2 = add_layer(l1, 2, 4, activation_function=tf.nn.relu)
    with tf.variable_scope("foo7"):
        l3 = add_layer(l2, 4, 3, activation_function=tf.nn.relu)
    with tf.variable_scope("foo8"):
        prediction = add_layer(l3,3, 2)#,activation_function=tf.nn.sigmoid)
    return prediction

# sampleSize=200;
# noisy_circles= datasets.make_moons(n_samples=sampleSize, noise=.05)
# X = noisy_circles[0]
# y=noisy_circles[1].reshape(1,sampleSize)
# Nz,Dz=X.shape
# sio.savemat('twoMoonData.mat', {'data': X,'inputK': y})
# print('down')

dataInput=sio.loadmat('twomoon1.mat')
# dataInput=sio.loadmat('manifoldSandW.mat')
# dataInput=sio.loadmat('manifoldSandWL.mat')
X=dataInput['data']
y=dataInput['inputK']
print(X.shape)
# print(X)
Nz,Dz=X.shape

# X=np.load("D:/ye/cars3/newData.npy")
# y=np.load("D:/ye/cars3/z.npy")
# Nz,Dz=X.shape

# x1,center=dataCombineX(X,70)
# x1.shape=(Nz,1)

hiden_size=1
xs = tf.placeholder(tf.float32, [None, Dz])
xs2 = tf.placeholder(tf.float32, [None, Dz])
dis=tf.placeholder(tf.float32, [1, None])
flag=tf.placeholder(tf.float32, [1, None])

with tf.variable_scope("model"):
    prediction=encoderVAE(xs, hiden_size)
    mean = prediction[:, :hiden_size]#+x1
    stddev = prediction[:, hiden_size:]*prediction[:, hiden_size:]
    epsilon = tf.random_normal([tf.shape(mean)[0], hiden_size])
    input_sample = mean + epsilon * stddev

    output=decoderVAE(input_sample,hiden_size)
with tf.variable_scope("model", reuse=True):
    prediction2=encoderVAE(xs2, hiden_size)
    mean2 = prediction2[:, :hiden_size]#+x1
    stddev2 = prediction2[:, hiden_size:]*prediction2[:, hiden_size:]
    epsilon = tf.random_normal([tf.shape(mean)[0], hiden_size])
    input_sample2 = mean2 + epsilon * stddev2

    output2=decoderVAE(input_sample2,hiden_size)

vae_loss = tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -2.0 * tf.log(stddev + 1e-8) - 1.0))
#rec_loss = tf.reduce_sum(-input_tensor * tf.log(output_tensor + 1e-8) -(1.0 - input_tensor) * tf.log(1.0 - output_tensor + 1e-8))
# rec_loss=tf.reduce_sum(tf.square(output-xs))

vae_loss2 = tf.reduce_sum(0.5 * (tf.square(mean2) + tf.square(stddev2) -2.0 * tf.log(stddev2 + 1e-8) - 1.0))
#rec_loss = tf.reduce_sum(-input_tensor * tf.log(output_tensor + 1e-8) -(1.0 - input_tensor) * tf.log(1.0 - output_tensor + 1e-8))
#rec_loss2=tf.reduce_sum(flag*tf.square(tf.reduce_sum(tf.square(output2-output),1)-dis))
rec_loss3=tf.reduce_sum(flag*tf.square(tf.reduce_sum(tf.square(mean2-mean),1)-dis))
rec_loss4=tf.reduce_sum(tf.reduce_sum(tf.square(output-xs)))

#for clustering
K=30;
D=hiden_size
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

k0=0
loss20=tf.constant(0.0)
tempU0=meanPlace[k0,:]#tf.to_float(tf.constant(uk[k,:]))
loss10=(tempU0-mean2)
tempB0=BPlace[:,:,k0]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro0=tf.matmul(loss10,tempB0)
tempA0=aPlace[k0]#tf.to_float(tf.constant(params['a'][k]))
loss3=tf.reduce_mean((tf.reshape(gammasPlace[:,k0],[-1,1])*(tempA0*tempPro0*loss10+tempA0*tempB0*stddev2)-0.5*tf.log(stddev2))/(1e5))

k1=tf.constant(1)
tempU1=meanPlace[k1,:]#tf.to_float(tf.constant(uk[k,:]))
loss11=(tempU1-mean2)
tempB1=BPlace[:,:,k1]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro1=tf.matmul(loss11,tempB1)
tempA1=aPlace[k1]#tf.to_float(tf.constant(params['a'][k]))
loss4=tf.reduce_mean((tf.reshape(gammasPlace[:,k1],[-1,1])*(tempA1*tempPro1*loss11+tempA1*tempB1*stddev2)-0.5*tf.log(stddev2))/(1e5))

k2=tf.constant(2)
tempU2=meanPlace[k2,:]#tf.to_float(tf.constant(uk[k,:]))
loss12=(tempU2-mean2)
tempB2=BPlace[:,:,k2]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro2=tf.matmul(loss12,tempB2)
tempA2=aPlace[k2]#tf.to_float(tf.constant(params['a'][k]))
loss5=tf.reduce_mean((tf.reshape(gammasPlace[:,k2],[-1,1])*(tempA2*tempPro2*loss12+tempA2*tempB2*stddev2)-0.5*tf.log(stddev2))/(1e5))

k3=tf.constant(3)
tempU3=meanPlace[k3,:]#tf.to_float(tf.constant(uk[k,:]))
loss13=(tempU3-mean2)
tempB3=BPlace[:,:,k3]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro3=tf.matmul(loss13,tempB3)
tempA3=aPlace[k3]#tf.to_float(tf.constant(params['a'][k]))
loss6=tf.reduce_mean((tf.reshape(gammasPlace[:,k3],[-1,1])*(tempA3*tempPro3*loss13+tempA3*tempB3*stddev2)-0.5*tf.log(stddev2))/(1e5))

k4=tf.constant(4)
tempU4=meanPlace[k4,:]#tf.to_float(tf.constant(uk[k,:]))
loss14=(tempU4-mean2)
tempB4=BPlace[:,:,k4]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro4=tf.matmul(loss14,tempB4)
tempA4=aPlace[k4]#tf.to_float(tf.constant(params['a'][k]))
loss7=tf.reduce_mean((tf.reshape(gammasPlace[:,k4],[-1,1])*(tempA4*tempPro4*loss14+tempA4*tempB4*stddev2)-0.5*tf.log(stddev2))/(1e5))

k5=tf.constant(5)
tempU5=meanPlace[k5,:]#tf.to_float(tf.constant(uk[k,:]))
loss15=(tempU5-mean2)
tempB5=BPlace[:,:,k5]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro5=tf.matmul(loss15,tempB5)
tempA5=aPlace[k5]#tf.to_float(tf.constant(params['a'][k]))
loss8=tf.reduce_mean((tf.reshape(gammasPlace[:,k5],[-1,1])*(tempA5*tempPro5*loss15+tempA5*tempB5*stddev2)-0.5*tf.log(stddev2))/(1e5))

k6=tf.constant(6)
tempU6=meanPlace[k6,:]#tf.to_float(tf.constant(uk[k,:]))
loss16=(tempU6-mean2)
tempB6=BPlace[:,:,k6]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro6=tf.matmul(loss16,tempB6)
tempA6=aPlace[k6]#tf.to_float(tf.constant(params['a'][k]))
loss9=tf.reduce_mean((tf.reshape(gammasPlace[:,k6],[-1,1])*(tempA6*tempPro6*loss16+tempA6*tempB6*stddev2)-0.5*tf.log(stddev2))/(1e5))

k7=tf.constant(7)
tempU7=meanPlace[k7,:]#tf.to_float(tf.constant(uk[k,:]))
loss17=(tempU7-mean2)
tempB7=BPlace[:,:,k7]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro7=tf.matmul(loss17,tempB7)
tempA7=aPlace[k7]#tf.to_float(tf.constant(params['a'][k]))
loss10=tf.reduce_mean((tf.reshape(gammasPlace[:,k7],[-1,1])*(tempA7*tempPro7*loss17+tempA7*tempB7*stddev2)-0.5*tf.log(stddev2))/(1e5))

k8=tf.constant(8)
tempU8=meanPlace[k8,:]#tf.to_float(tf.constant(uk[k,:]))
loss18=(tempU8-mean2)
tempB8=BPlace[:,:,k8]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro8=tf.matmul(loss18,tempB8)
tempA8=aPlace[k8]#tf.to_float(tf.constant(params['a'][k]))
loss11=tf.reduce_mean((tf.reshape(gammasPlace[:,k8],[-1,1])*(tempA8*tempPro8*loss18+tempA8*tempB8*stddev2)-0.5*tf.log(stddev2))/(1e5))

k9=tf.constant(9)
tempU9=meanPlace[k9,:]#tf.to_float(tf.constant(uk[k,:]))
loss19=(tempU9-mean2)
tempB9=BPlace[:,:,k9]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro9=tf.matmul(loss19,tempB9)
tempA9=aPlace[k9]#tf.to_float(tf.constant(params['a'][k]))
loss12=tf.reduce_mean((tf.reshape(gammasPlace[:,k9],[-1,1])*(tempA9*tempPro9*loss19+tempA9*tempB9*stddev2)-0.5*tf.log(stddev2))/(1e5))

k10=tf.constant(10)
tempU10=meanPlace[k10,:]#tf.to_float(tf.constant(uk[k,:]))
loss110=(tempU10-mean2)
tempB10=BPlace[:,:,k10]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro10=tf.matmul(loss110,tempB9)
tempA10=aPlace[k10]#tf.to_float(tf.constant(params['a'][k]))
loss13=tf.reduce_mean((tf.reshape(gammasPlace[:,k10],[-1,1])*(tempA10*tempPro10*loss110+tempA10*tempB10*stddev2)-0.5*tf.log(stddev2))/(1e5))

k11=tf.constant(11)
tempU11=meanPlace[k11,:]#tf.to_float(tf.constant(uk[k,:]))
loss111=(tempU11-mean2)
tempB11=BPlace[:,:,k11]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro11=tf.matmul(loss111,tempB11)
tempA11=aPlace[k11]#tf.to_float(tf.constant(params['a'][k]))
loss14=tf.reduce_mean((tf.reshape(gammasPlace[:,k11],[-1,1])*(tempA11*tempPro11*loss111+tempA11*tempB11*stddev2)-0.5*tf.log(stddev2))/(1e5))

k12=tf.constant(12)
tempU12=meanPlace[k12,:]#tf.to_float(tf.constant(uk[k,:]))
loss112=(tempU12-mean2)
tempB12=BPlace[:,:,k12]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro12=tf.matmul(loss112,tempB12)
tempA12=aPlace[k12]#tf.to_float(tf.constant(params['a'][k]))
loss15=tf.reduce_mean((tf.reshape(gammasPlace[:,k12],[-1,1])*(tempA12*tempPro12*loss112+tempA12*tempB12*stddev2)-0.5*tf.log(stddev2))/(1e5))

k13=tf.constant(13)
tempU13=meanPlace[k13,:]#tf.to_float(tf.constant(uk[k,:]))
loss113=(tempU13-mean2)
tempB13=BPlace[:,:,k13]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro13=tf.matmul(loss113,tempB13)
tempA13=aPlace[k13]#tf.to_float(tf.constant(params['a'][k]))
loss16=tf.reduce_mean((tf.reshape(gammasPlace[:,k13],[-1,1])*(tempA13*tempPro13*loss113+tempA13*tempB13*stddev2)-0.5*tf.log(stddev2))/(1e5))

k14=tf.constant(14)
tempU14=meanPlace[k14,:]#tf.to_float(tf.constant(uk[k,:]))
loss114=(tempU14-mean2)
tempB14=BPlace[:,:,k14]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro14=tf.matmul(loss114,tempB14)
tempA14=aPlace[k14]#tf.to_float(tf.constant(params['a'][k]))
loss17=tf.reduce_mean((tf.reshape(gammasPlace[:,k14],[-1,1])*(tempA14*tempPro14*loss114+tempA14*tempB14*stddev2)-0.5*tf.log(stddev2))/(1e5))

k15=tf.constant(15)
tempU15=meanPlace[k15,:]#tf.to_float(tf.constant(uk[k,:]))
loss115=(tempU15-mean2)
tempB15=BPlace[:,:,k15]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro15=tf.matmul(loss115,tempB15)
tempA15=aPlace[k15]#tf.to_float(tf.constant(params['a'][k]))
loss18=tf.reduce_mean((tf.reshape(gammasPlace[:,k15],[-1,1])*(tempA15*tempPro15*loss115+tempA15*tempB15*stddev2)-0.5*tf.log(stddev2))/(1e5))

k16=tf.constant(16)
tempU16=meanPlace[k16,:]#tf.to_float(tf.constant(uk[k,:]))
loss116=(tempU16-mean2)
tempB16=BPlace[:,:,k16]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro16=tf.matmul(loss116,tempB16)
tempA16=aPlace[k16]#tf.to_float(tf.constant(params['a'][k]))
loss19=tf.reduce_mean((tf.reshape(gammasPlace[:,k16],[-1,1])*(tempA16*tempPro16*loss116+tempA16*tempB16*stddev2)-0.5*tf.log(stddev2))/(1e5))

k17=tf.constant(17)
tempU17=meanPlace[k17,:]#tf.to_float(tf.constant(uk[k,:]))
loss117=(tempU17-mean2)
tempB17=BPlace[:,:,k17]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro17=tf.matmul(loss117,tempB17)
tempA17=aPlace[k17]#tf.to_float(tf.constant(params['a'][k]))
loss20=tf.reduce_mean((tf.reshape(gammasPlace[:,k17],[-1,1])*(tempA17*tempPro17*loss117+tempA17*tempB17*stddev2)-0.5*tf.log(stddev2))/(1e5))

k18=tf.constant(18)
tempU18=meanPlace[k18,:]#tf.to_float(tf.constant(uk[k,:]))
loss118=(tempU18-mean2)
tempB18=BPlace[:,:,k18]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro18=tf.matmul(loss118,tempB18)
tempA18=aPlace[k18]#tf.to_float(tf.constant(params['a'][k]))
loss21=tf.reduce_mean((tf.reshape(gammasPlace[:,k18],[-1,1])*(tempA18*tempPro18*loss118+tempA18*tempB18*stddev2)-0.5*tf.log(stddev2))/(1e5))

k19=tf.constant(19)
tempU19=meanPlace[k19,:]#tf.to_float(tf.constant(uk[k,:]))
loss119=(tempU19-mean2)
tempB19=BPlace[:,:,k19]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro19=tf.matmul(loss119,tempB19)
tempA19=aPlace[k19]#tf.to_float(tf.constant(params['a'][k]))
loss22=tf.reduce_mean((tf.reshape(gammasPlace[:,k19],[-1,1])*(tempA19*tempPro19*loss119+tempA19*tempB19*stddev2)-0.5*tf.log(stddev2))/(1e5))

k20=tf.constant(20)
tempU20=meanPlace[k20,:]#tf.to_float(tf.constant(uk[k,:]))
loss120=(tempU20-mean2)
tempB20=BPlace[:,:,k20]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro20=tf.matmul(loss120,tempB20)
tempA20=aPlace[k20]#tf.to_float(tf.constant(params['a'][k]))
loss23=tf.reduce_mean((tf.reshape(gammasPlace[:,k20],[-1,1])*(tempA20*tempPro20*loss120+tempA20*tempB20*stddev2)-0.5*tf.log(stddev2))/(1e5))

k21=tf.constant(21)
tempU21=meanPlace[k21,:]#tf.to_float(tf.constant(uk[k,:]))
loss121=(tempU21-mean2)
tempB21=BPlace[:,:,k21]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro21=tf.matmul(loss121,tempB21)
tempA21=aPlace[k21]#tf.to_float(tf.constant(params['a'][k]))
loss24=tf.reduce_mean((tf.reshape(gammasPlace[:,k21],[-1,1])*(tempA21*tempPro21*loss121+tempA21*tempB21*stddev2)-0.5*tf.log(stddev2))/(1e5))

k22=tf.constant(22)
tempU22=meanPlace[k22,:]#tf.to_float(tf.constant(uk[k,:]))
loss122=(tempU22-mean2)
tempB22=BPlace[:,:,k22]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro22=tf.matmul(loss122,tempB22)
tempA22=aPlace[k22]#tf.to_float(tf.constant(params['a'][k]))
loss25=tf.reduce_mean((tf.reshape(gammasPlace[:,k22],[-1,1])*(tempA22*tempPro22*loss122+tempA22*tempB22*stddev2)-0.5*tf.log(stddev2))/(1e5))

k23=tf.constant(23)
tempU23=meanPlace[k23,:]#tf.to_float(tf.constant(uk[k,:]))
loss123=(tempU23-mean2)
tempB23=BPlace[:,:,k23]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro23=tf.matmul(loss123,tempB23)
tempA23=aPlace[k23]#tf.to_float(tf.constant(params['a'][k]))
loss26=tf.reduce_mean((tf.reshape(gammasPlace[:,k23],[-1,1])*(tempA23*tempPro23*loss123+tempA23*tempB23*stddev2)-0.5*tf.log(stddev2))/(1e5))

k24=tf.constant(24)
tempU24=meanPlace[k24,:]#tf.to_float(tf.constant(uk[k,:]))
loss124=(tempU24-mean2)
tempB24=BPlace[:,:,k24]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro24=tf.matmul(loss124,tempB24)
tempA24=aPlace[k24]#tf.to_float(tf.constant(params['a'][k]))
loss27=tf.reduce_mean((tf.reshape(gammasPlace[:,k24],[-1,1])*(tempA24*tempPro24*loss124+tempA24*tempB24*stddev2)-0.5*tf.log(stddev2))/(1e5))

k25=tf.constant(25)
tempU25=meanPlace[k25,:]#tf.to_float(tf.constant(uk[k,:]))
loss125=(tempU25-mean2)
tempB25=BPlace[:,:,k25]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro25=tf.matmul(loss125,tempB25)
tempA25=aPlace[k25]#tf.to_float(tf.constant(params['a'][k]))
loss28=tf.reduce_mean((tf.reshape(gammasPlace[:,k25],[-1,1])*(tempA25*tempPro25*loss125+tempA25*tempB25*stddev2)-0.5*tf.log(stddev2))/(1e5))

k26=tf.constant(26)
tempU26=meanPlace[k26,:]#tf.to_float(tf.constant(uk[k,:]))
loss126=(tempU26-mean2)
tempB26=BPlace[:,:,k26]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro26=tf.matmul(loss126,tempB26)
tempA26=aPlace[k26]#tf.to_float(tf.constant(params['a'][k]))
loss29=tf.reduce_mean((tf.reshape(gammasPlace[:,k26],[-1,1])*(tempA26*tempPro26*loss126+tempA26*tempB26*stddev2)-0.5*tf.log(stddev2))/(1e5))

k27=tf.constant(27)
tempU27=meanPlace[k27,:]#tf.to_float(tf.constant(uk[k,:]))
loss127=(tempU27-mean2)
tempB27=BPlace[:,:,k27]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro27=tf.matmul(loss127,tempB27)
tempA27=aPlace[k27]#tf.to_float(tf.constant(params['a'][k]))
loss30=tf.reduce_mean((tf.reshape(gammasPlace[:,k27],[-1,1])*(tempA27*tempPro27*loss127+tempA27*tempB27*stddev2)-0.5*tf.log(stddev2))/(1e5))

k29=tf.constant(29)
tempU29=meanPlace[k29,:]#tf.to_float(tf.constant(uk[k,:]))
loss129=(tempU29-mean2)
tempB29=BPlace[:,:,k29]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro29=tf.matmul(loss129,tempB29)
tempA29=aPlace[k29]#tf.to_float(tf.constant(params['a'][k]))
loss31=tf.reduce_mean((tf.reshape(gammasPlace[:,k29],[-1,1])*(tempA29*tempPro29*loss129+tempA29*tempB29*stddev2)-0.5*tf.log(stddev2))/(1e5))

k28=tf.constant(28)
tempU28=meanPlace[k28,:]#tf.to_float(tf.constant(uk[k,:]))
loss128=(tempU28-mean2)
tempB28=BPlace[:,:,k28]#tf.to_float(tf.constant(wk[:,:,k]))
tempPro28=tf.matmul(loss128,tempB28)
tempA28=aPlace[k28]#tf.to_float(tf.constant(params['a'][k]))
loss32=tf.reduce_mean((tf.reshape(gammasPlace[:,k28],[-1,1])*(tempA28*tempPro28*loss128+tempA28*tempB28*stddev2)-0.5*tf.log(stddev2))/(1e5))

lossFinal1 = 0.1*(loss0+loss1+loss3+loss4+loss5+loss6+loss7+loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss15+loss16+loss17+loss18+loss19+loss20+loss11+loss21+loss22+loss23+loss24+loss25+loss26+loss27+loss28+loss29+loss30+loss31+loss31+loss32)
lossFinal2 =  rec_loss3+rec_loss4#rec_loss2
finalLoss=lossFinal1+lossFinal2
train_step1 = tf.train.GradientDescentOptimizer(0.0000001).minimize(lossFinal1+lossFinal2)
train_step2 = tf.train.GradientDescentOptimizer(0.0001).minimize(lossFinal2)

init = tf.global_variables_initializer()

run_config = tf.ConfigProto()
run_config.gpu_options.allow_growth=True
run_config.gpu_options.per_process_gpu_memory_fraction = 1/10
sess = tf.Session(config=run_config)
sess.run(init)

#compute k nearest neighorhood
flagReal = kneighbors_graph(X, 10, mode='connectivity', include_self=True)
disReal = kneighbors_graph(X, 10, mode='distance', include_self=True)
flagReal=flagReal.toarray()
flagRealPure=flagReal
disReal=disReal.toarray()


flagReal=(flagReal-1)
flagReal=flagReal*flagReal
disReal=100*flagReal+flagRealPure*10
flagReal=flagRealPure+flagReal

flagReal2 = kneighbors_graph(X, 5, mode='connectivity', include_self=True)
disReal2 = kneighbors_graph(X, 5, mode='distance', include_self=True)
flagReal2=flagReal2.toarray()
flagReal2Pure=flagReal2
disReal2=disReal2.toarray()

flagReal2=(flagReal2-1)
flagReal2=flagReal2*flagReal2
disReal2=30*flagReal2+flagReal2Pure*10
flagReal2=flagReal2+flagReal2Pure

posMean=sess.run(mean,feed_dict={xs: X})
params,gammas = vdpmm_init(posMean,K)
paramsGaussian,gammasGaussian = vdpmm_init(X,K)

print("begin optimizing the network")

lambdaPos=0
numits=1;
maxits=100;
while numits < maxits:

    lossReal=0
    for i in range(Nz):
        Xtemp2=np.matlib.repmat(X[i,:], Nz, 1)
        Xtemp1=X
        disTemp=disReal[i,:].reshape(1,Nz);
        flagTemp=flagReal[i,:].reshape(1,Nz);
        sess.run(train_step1, feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp,xs: X,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # posGammas=sess.run(train_step2, feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp,xs: X,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})

        # sess.run(train_step,feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp})
        lossReal=lossReal+sess.run(finalLoss,feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp,xs: X,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
    print(lossReal);
    posMean=sess.run(mean,feed_dict={xs: X})
    posCov=sess.run(stddev,feed_dict={xs: X})
    paramsGaussian = vdpmm_maximizePlusGaussian(X, paramsGaussian,  (lambdaPos) *gammas)
    posGaussian = vdpmm_expectationPlusGaussian(X, paramsGaussian)
    params = vdpmm_maximizeCNN(posMean,params,(1 - lambdaPos) *gammas,posCov);
    gammas=vdpmm_expectationCNNKnearset(posMean,params,lambdaPos,(lambdaPos)*posGaussian)
    numits=numits+1;
print("begin optimizing the network refining stage")
numits=1;
maxits=10;
while numits < maxits:
    lossReal=0
    posMean=sess.run(mean,feed_dict={xs: X})
    posCov=sess.run(stddev,feed_dict={xs: X})
    for i in range(Nz):
        Xtemp2=np.matlib.repmat(X[i,:], Nz, 1)
        Xtemp1=X
        disTemp2=disReal2[i,:].reshape(1,Nz);
        flagTemp2=flagReal2[i,:].reshape(1,Nz);
        posGammas=sess.run(train_step1, feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp2,flag:flagTemp2,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # posGammas=sess.run(train_step2, feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp2,flag:flagTemp2,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})
        # sess.run(train_step,feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp})
        # sess.run(train_step,feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp,flag:flagTemp})
        lossReal=lossReal+sess.run(finalLoss,feed_dict={xs:Xtemp1,xs2:Xtemp2,dis:disTemp2,flag:flagTemp2,BPlace:params['B'],aPlace:params['a'].reshape(K,1),meanPlace:params['mean'],gammasPlace:gammas})

    print(lossReal);
    paramsGaussian = vdpmm_maximizePlusGaussian(X, paramsGaussian,  ( lambdaPos) *gammas)
    posGaussian = vdpmm_expectationPlusGaussian(X, paramsGaussian)
    params = vdpmm_maximizeCNN(posMean,params,(1 - lambdaPos)*gammas,posCov);
    gammas=vdpmm_expectationCNNKnearset(posMean,params,lambdaPos,(lambdaPos)*posGaussian)
    numits=numits+1;

newData=sess.run(mean,feed_dict={xs:X})
print(newData)

print(params['mean'])
print(gammas)
print(y.shape)
colorStore='rgbyck'
plt.figure(1)
for i in range(Nz):
    cho=(np.mod(y[i]+1,6))[0]
    plt.scatter(newData[i,0],newData[i,0],color=colorStore[cho])

temp=np.max(gammas,axis=1)
temp.shape=(Nz,1)
index1=np.where(temp==gammas)
print(index1)
plt.figure(2)
for i in range(Nz):
    cho=np.mod(index1[1][i],6)
    plt.scatter(X[i,0],X[i,1],color=colorStore[cho])

# fig=plt.figure(2)
# ax1=fig.add_subplot(111,projection='3d')
# for i in range(Nz):
#     cho=np.mod(index1[1][i]+1,6)
#     # plt.scatter(posMean[i,0],posMean[i,0],color=colorStore[cho])
#     ax1.scatter(X[i,0],X[i,1],X[i,2],color=colorStore[cho])

plt.show()

preLabel=list(index1[1])
srcLabel=(y.T)[0]
print(preLabel)
print(srcLabel)

print(metrics.adjusted_mutual_info_score(preLabel, srcLabel))
print(metrics.adjusted_rand_score(preLabel, srcLabel))

print(np.unique(index1[1]))
print(lambdaPos)

sio.savemat('dataNeural.mat',{'data':newData, 'inputK':y})