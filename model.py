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
from keras.layers import Input, add, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Add,Concatenate, concatenate, Dropout
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras


BATCH_SIZE = 2
class Model:
	def _conv_layer(self, name, input_var,KERNEL_SIZE, stride, in_channels, out_channels, options = {}):
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

	def RDBlocks(self, x, name, g):
		## 6 layers of RDB block
		## this thing need to be in a damn loop for more customisability
		li = [x]
		count = 2
		x_input = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
						 name=name + '_conv1_1')(x)
		pas = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
					 name=name + '_conv1')(x)

		for i in range(2, count + 1):
			li.append(pas)
			out = Concatenate(axis=self.channel_axis)(li)  # conctenated out put
			pas = Conv2D(filters=g, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
						 name=name + '_conv' + str(i))(out)

		# feature extractor from the dense net
		li.append(pas)
		out = Concatenate(axis=self.channel_axis)(li)
		feat = Conv2D(filters=g, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu',
					  name=name + '_Local_Conv')(out)

		feat = Add()([feat, x_input])
		return feat


	def __init__(self, big=False):
		tf.reset_default_graph()
		self.channel_axis = 3

		self.is_training = tf.placeholder(tf.bool)
		if big:
			self.inputs = tf.placeholder(tf.float32, [None, 2048, 2048, 3])
			self.targets = tf.placeholder(tf.float32, [None, 2048, 2048, 1])
		else:
			self.inputs = tf.placeholder(tf.float32, [None, 256, 256, 3])
			self.targets = tf.placeholder(tf.float32, [None, 256, 256, 1])
		self.learning_rate = tf.placeholder(tf.float32)

		self.dropout_factor = tf.to_float(self.is_training) * 0.3
		## encoder
		self.layer1 = self._conv_layer('layer1', self.inputs, 3, 1, 3, 32, {'activation':'relu','batchnorm': True})#256*256
		#self.layer2 = self._conv_layer('layer2', self.layer1, 3, 1, 16, 32, {'batchnorm': False})
		self.layer3 = self._conv_layer('layer3', self.layer1, 3, 1, 32, 32, {'batchnorm': True})
		#self.layer4_inputs = tf.add(self.layer3, self.layer1)
		self.layer4 = self.RDBlocks(self.layer3, 'RDB1', 32)#256*256
		self.layer5 = self._conv_layer('layer5', self.layer4, 3, 2, 32, 64, {'batchnorm': True})#128*128
		self.layer6 = self.RDBlocks(self.layer5, 'RDB2', 64)#128*128
		self.layer7 = self._conv_layer('layer7', self.layer6, 3, 2, 64, 128, {'batchnorm': True})#64*64
		self.layer8 = self.RDBlocks(self.layer7, 'RDB3', 128)#64*64
		self.layer9 = self._conv_layer('layer9', self.layer8, 3, 2, 128, 256, {'batchnorm': True})#32*32

		## bridge
		self.layer10 = self.RDBlocks(self.layer9, 'RDB4', 256)

		## decoder
		self.layer11 = self._conv_layer('layer11', self.layer10, 3, 2, 256, 128, {'transpose': True})
		self.layer12_inputs = tf.concat([self.layer11, self.layer8], axis=3)
		self.layer12 = self.RDBlocks(self.layer12_inputs, 'RDB5', 128)
		self.layer13 = self._conv_layer('layer13', self.layer12, 3, 2, 128, 64, {'transpose': True})
		self.layer14_inputs = tf.concat([self.layer13, self.layer6], axis=3)
		self.layer14 = self.RDBlocks(self.layer14_inputs, 'RDB6', 64)
		self.layer15 = self._conv_layer('layer15', self.layer14, 3, 2, 64, 32, {'transpose': True})
		self.layer16_inputs = tf.concat([self.layer15, self.layer4], axis=3)
		self.layer16 = self.RDBlocks(self.layer16_inputs, 'RDB7',  64)
		#self.layer17 = self._conv_layer('layer17', self.layer16, 3, 1, 64, 64, { 'batchnorm': False})

		self.pre_outputs = self._conv_layer('pre_outputs', self.layer16, 1, 1, 64, 2, {'activation': 'none', 'batchnorm': False}) # -> 256x256x2

		self.outputs = tf.nn.softmax(self.pre_outputs)[:, :, :, 0]
		self.labels = tf.concat([self.targets, 1 - self.targets], axis=3)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.pre_outputs))

		with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

		self.init_op = tf.initialize_all_variables()
		self.saver = tf.train.Saver(max_to_keep=None)
