# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:38:13 2017

@author: ye
"""

from __future__ import absolute_import, division, print_function

import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neighbors import kneighbors_graph

#from progressbar import ETA, Bar, Percentage, ProgressBar


from gan import GAN



if __name__ == "__main__":
    data_directory = os.path.join("", "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)
    model = GAN(128, 128, 1e-2)

    # orderedImage=np.load("orderedImage6.npy")
    for epoch in range(1000):
        training_loss = 0.0
        batch_size=128;


        for i in range(40):
            images, _ = mnist.train.next_batch(128)
            flagReal = kneighbors_graph(images, 5, mode='connectivity', include_self=True)
            disReal = kneighbors_graph(images, 5, mode='distance', include_self=True)
            flagReal = flagReal.toarray()
            disReal = disReal.toarray()
            
            flagReal2 = (flagReal - 1)
            flagReal2 = flagReal2 * flagReal2
            disReal = 100 * flagReal2 + flagReal * 1
            for i in range(batch_size):
                Xtemp2 = np.matlib.repmat(images[i, :], batch_size, 1)
                disTemp = disReal[i, :].reshape(1, batch_size);
                flagTemp = flagReal[i, :].reshape(1, batch_size);
            # images=orderedImage[i*128:(i+1)*128,:]
                loss_value = model.update_params(images,images,flagTemp,disTemp)
                training_loss += loss_value
        training_loss = training_loss /(1000 * 128)
        print("Loss %f" % training_loss)
        model.generate_and_save_images(
            128, "")
