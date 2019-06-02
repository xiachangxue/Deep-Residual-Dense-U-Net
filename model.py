import numpy
import tensorflow as tf
import os
import os.path
import random
import math
import time
from PIL import Image
import keras
from keras.models import *
from keras.layers import Input, Conv2D,  UpSampling2D, BatchNormalization, Activation, add, concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint

BATCH_SIZE = 2
KERNEL_SIZE = 3

def res_block(x, nb_filters, strides):
    res_path = BatchNormalization()(x)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[0], kernel_size=(3, 3), padding='same', strides=strides[0])(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = Activation(activation='relu')(res_path)
    res_path = Conv2D(filters=nb_filters[1], kernel_size=(3, 3), padding='same', strides=strides[1])(res_path)

    shortcut = Conv2D(nb_filters[1], kernel_size=(1, 1), strides=strides[0])(x)
    shortcut = BatchNormalization()(shortcut)

    res_path = add([shortcut, res_path])
    return res_path

class Model:
	def _conv_layer(self, name, input_var, stride, in_channels, out_channels, options = {}):
		activation = options.get('activation', 'relu')
		dropout = options.get('dropout', None)
		padding = options.get('padding', 'SAME')
		batchnorm = options.get('batchnorm', True)
		transpose = options.get('transpose', False)

		with tf.variable_scope(name) as scope:
			if not transpose:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, in_channels, out_channels]
			else:
				filter_shape = [KERNEL_SIZE, KERNEL_SIZE, out_channels, in_channels]
			kernel = tf.get_variable(
				'weights',
				shape=filter_shape,
				initializer=tf.truncated_normal_initializer(stddev=math.sqrt(2.0 / KERNEL_SIZE / KERNEL_SIZE / in_channels)),
				dtype=tf.float32
			)
			biases = tf.get_variable(
				'biases',
				shape=[out_channels],
				initializer=tf.constant_initializer(0.0),
				dtype=tf.float32
			)
			if not transpose:
				output = tf.nn.bias_add(
					tf.nn.conv2d(
						input_var,
						kernel,
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			else:
				batch = tf.shape(input_var)[0]
				side = tf.shape(input_var)[1]
				output = tf.nn.bias_add(
					tf.nn.conv2d_transpose(
						input_var,
						kernel,
						[batch, side * stride, side * stride, out_channels],
						[1, stride, stride, 1],
						padding=padding
					),
					biases
				)
			if batchnorm:
				output = tf.contrib.layers.batch_norm(output, center=True, scale=True, is_training=self.is_training, decay=0.99)
			if dropout is not None:
				output = tf.nn.dropout(output, keep_prob=1-dropout)

			if activation == 'relu':
				return tf.nn.relu(output, name=scope.name)
			elif activation == 'sigmoid':
				return tf.nn.sigmoid(output, name=scope.name)
			elif activation == 'none':
				return output
			else:
				raise Exception('invalid activation {} specified'.format(activation))

	def __init__(self, big=False):
		tf.reset_default_graph()

		self.is_training = tf.placeholder(tf.bool)
		if big:
			self.inputs = tf.placeholder(tf.float32, [None, 2048, 2048, 3])
			self.targets = tf.placeholder(tf.float32, [None, 2048, 2048, 1])
		else:
			self.inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
			self.targets = tf.placeholder(tf.float32, [None, 256, 256, 1])
		self.learning_rate = tf.placeholder(tf.float32)

		self.dropout_factor = tf.to_float(self.is_training) * 0.3

		to_decoder = []

		## encoder
		main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(self.inputs)
		main_path = BatchNormalization()(main_path)
		main_path = Activation(activation='relu')(main_path)

		main_path = Conv2D(filters=64, kernel_size=(3, 3), padding='same', strides=(1, 1))(main_path)

		shortcut = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(main_path)
		shortcut = BatchNormalization()(shortcut)

		main_path = add([shortcut, main_path])
		# first branching to decoder
		to_decoder.append(main_path)

		main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)])
		to_decoder.append(main_path)

		main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)])
		to_decoder.append(main_path)

		## res_block
		main_path = res_block(to_decoder[2], [512, 512], [(2, 2), (1, 1)])

		##decoder
		main_path = UpSampling2D(size=(2, 2))(main_path)
		main_path = concatenate([main_path, to_decoder[2]], axis=3)
		main_path = res_block(main_path, [256, 256], [(1, 1), (1, 1)])

		main_path = UpSampling2D(size=(2, 2))(main_path)
		main_path = concatenate([main_path, to_decoder[1]], axis=3)
		main_path = res_block(main_path, [128, 128], [(1, 1), (1, 1)])

		main_path = UpSampling2D(size=(2, 2))(main_path)
		main_path = concatenate([main_path, to_decoder[0]], axis=3)
		main_path = res_block(main_path, [64, 64], [(1, 1), (1, 1)])

		self.pre_outputs = self._conv_layer('pre_outputs', main_path, 1, 64, 2, {'activation': 'none', 'batchnorm': False}) # -> 256x256x2

		self.outputs = tf.nn.softmax(self.pre_outputs)[:, :, :, 0]
		self.labels = tf.concat([self.targets, 1 - self.targets], axis=3)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.pre_outputs))

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
