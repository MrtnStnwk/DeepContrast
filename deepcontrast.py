# Inspired by / code reused from:
# Simon Karlsson https://github.com/simontomaskarlsson/CycleGAN-Keras
# Fabian Isensee (2017)
# many others


import tensorflow as tf
tf.compat.v1.disable_eager_execution()
gpus = tf.config.experimental.list_physical_devices('GPU') 

# Currently, memory growth needs to be the same across GPUs
for gpu in gpus:
	tf.config.experimental.set_memory_growth(gpu, True)
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

from tensorflow.python import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Input, Conv2D, Activation, Concatenate, UpSampling2D, BatchNormalization, Conv3D, UpSampling3D, SpatialDropout2D, Add,SpatialDropout3D,Dropout,InputLayer,LeakyReLU, AveragePooling2D, AveragePooling3D
from tensorflow.python.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, model_from_json, clone_model
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras.regularizers import l2, l1
from tensorflow.keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
from tensorflow.python.keras.utils.data_utils import OrderedEnqueuer
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization    # ->> instancenormalization is moved from keras_contrib to tensorflow_addons. This only works with tensorflow 2.1.0 at the moment 
from tensorflow_addons.layers import InstanceNormalization

from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score

import optparse, random, datetime, time, json, csv, pickle, sys, os, re, scipy

from collections import OrderedDict
import numpy as np
import math
from scipy.spatial import distance
from PIL import Image
import warnings

import matplotlib.pyplot as plt
import nibabel as nib

import load_data_mmap as load_data
import losses, generators

from data_generator import data_generator

# https://github.com/hanwen0529/GAN-pix2pix-Keras/blob/master/pix2pix.py#L29
class DeepContras():
	def __init__(self,augment_opts,lr_D=2e-4, lr_G=0.001, loss_G='mae', network="", nfilters=64, normalization=None, final_activation="linear", batch_size = 1, 
		image_shapeA=(256*1,256*1,1),image_shapeB=(256*1,256*1,1),date_time_string_addition='_pix2pix', image_folder='',
		trainmode='image', scaling="mask_variance", pre_norm = False, 
		use_gan = False, workers=1,
		save_interval=1,save_interval_ntest=0,
		applyFCN=False,
		initG=None): 

		self.ndims = len(image_shapeA)-1

		self.trainmode = trainmode

		self.img_shapeA = image_shapeA
		self.channelsA = self.img_shapeA[-1]
	
		self.img_shapeB = image_shapeB
		self.channelsB = self.img_shapeB[-1]
		
		self.nii = True				# this is always true (method to run from images is not implemented anymore). 
		self.use_gan = use_gan

		# hyper parameters
		self.lambda_1 = 100.0 # loss weight A_2_B
		self.lambda_D = 1.0
		self.learning_rate_D = lr_G
		self.learning_rate_G = lr_G
		self.network = network
		
		if self.use_gan:
			self.beta_1 = 0.5
		else:
			self.beta_1 = 0.9
		self.beta_2 = 0.999

		self.batch_size = batch_size
		self.epochs = 200			# this is the number of epochs. 
		self.save_interval = save_interval#5			# this is the save interval (when images and loss function are stored in the /niis folder
		self.save_interval_ntest = save_interval_ntest

		self.calculate_distmap = False
		self.alpha = K.variable(value=0.5)

		self.scaling = scaling			# scaling is the scaling that is done after loading (normally: z-score within mask)
		self.pre_norm = pre_norm 		# pre norm is only implemented for 2d. It performs within mask z-transformation after data augmentation, before putting patches to GPU. 
		self.workers = workers

		self.applyFCN = applyFCN
		self.initG = initG

		self.cpu_merge = False
		self.strategy = tf.distribute.experimental.CentralStorageStrategy() #MirroredStrategy()

		self.reg = None #l2(0.01)

		print(loss_G)
		if loss_G == 'mae':
			self.loss_G = 'mae'

		else:
			warnings.warn('No valid loss function')
			exit()

		self.nfilters = nfilters
		self.normalization = normalization
		self.final_activation = final_activation

		self.augment_opts = augment_opts

		# PatchGAN - if false the discriminator learning rate should be decreased
		self.use_patchgan = True

		# Fetch data during training instead of pre caching all images - might be necessary for large datasets
		self.use_data_generator = True
		self.use_val_data_generator = False

		# GENERATE STORAGE FOLDER NAME
		# Used as storage folder name
		self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition


		if self.trainmode == 'patch':
			sz = self.img_shapeA[0] 
			self.testString = self.date_time + '_' + image_folder + '_ndims' + str(self.ndims) + '_sc' + scaling + '_lossG' + loss_G + \
			'_batch' + str(self.batch_size) + '_netw' + self.network + '_nfilters' + str(self.nfilters) + \
			'_norm' + str(self.normalization) + '_prenorm' + str(self.pre_norm) + '_patch' + str(sz) + '_lrG' + str(self.learning_rate_G) + \
			'_augmentspatial' + str(int(self.augment_opts['spatial'])) + '_usegan' + str(int(self.use_gan)) + \
			'applyFCN' + str(int(self.applyFCN))
		
		print(self.testString)
	
		# optimizer:
		self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)#,clipvalue=0.0001)
		self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)#,clipvalue=0.0001)
	
		# Number of filters in the first layer of G and D
		self.df = self.nfilters
		self.gf = self.nfilters
		
		if self.use_gan:
			# Build and compile the discriminator
			self.discriminator = self.build_discriminator(normalization=self.normalization,reg=self.reg) 

			print('DISCRIMINATOR SUMMARY')
			self.discriminator.summary()

			self.parallel_discriminator = self.discriminator

			print('PARALLEL DISCRIMINATOR SUMMARY')
			self.parallel_discriminator.summary()

			if True:
				self.parallel_discriminator.compile(loss='binary_crossentropy',
					optimizer=self.opt_D,
					metrics=['accuracy'], loss_weights=[0.5])
		
		#-------------------------
		# Construct Computational
		#   Graph of Generator
	    	#-------------------------

		# Build the generator
		print(self.network)
		if self.network.startswith("isensee"):
			digits = re.findall(r'\d+',self.network) #[int(i) for i in self.network.split() if i.isdigit()]
			print('Using isensee with depth ', digits[0], ' and ', digits[1], 'segmentation layers')

			if not self.applyFCN:

				self.generator = generators.build_generator_isensee2017_3d(self.img_shapeB, n_base_filters=self.nfilters, n_labels=self.channelsA, \
					activation_name=self.final_activation, \
					depth=int(digits[0]),n_segmentation_levels=int(digits[1]), \
					normalization=self.normalization,reg=self.reg,strategy=None)
			else:
				with open('saved_models/20200308-100636_pix2pix/model_1_model_epoch_350.json') as json_file:
					data = json.load(json_file)
					self.generator = model_from_json(data)
		else:
			print("No valid network specified")
			exit()


		if self.initG is not None:
			print('Loading generator weights from',self.initG)
			self.generator.load_weights(self.initG+'.hdf5')	

			if self.applyFCN:
				# ======= Data ==========
		        
				data = load_data.load_data(nr_of_channelsA=self.channelsA,
					nr_of_channelsB=self.channelsB,
					nr_A_train_imgs=1,
					nr_B_train_imgs=1,
					batch_size=self.batch_size,
					subfolder=image_folder,
					nii=self.nii,
					ndim=self.ndims,
					scaling=scaling,
					final_activation=self.final_activation,
					distmap=self.calculate_distmap)
				
				self.A_train = data["trainA_images"]
				self.A_train_masks = data["trainA_masks"]
				self.B_train = data["trainB_images"]
				self.A_test = data["testA_images"]
				self.A_test_masks = data['testA_masks']
				self.B_test = data["testB_images"]
				self.testB_image_names = data["testB_image_names"]
				if self.trainmode == 'image':
					self.testA_image_z = data["testA_image_z"]
					self.testB_image_z = data["testB_image_z"]

				self.saveNiisPatch3D(epoch=0,mode='fcn')
				exit()

	
		print('GENERATOR SUMMARY')
		self.generator.summary()

		self.parallel_generator = self.generator
		
		print('PARALLEL GENERATOR SUMMARY')
		self.parallel_generator.summary()

		# Input images and their conditioning images
		img_A = Input(shape=self.img_shapeA)
		img_B = Input(shape=self.img_shapeB)

		# By conditioning on B generate a fake version of A
		fake_A = self.parallel_generator(img_B)

		if self.use_gan:
			# For the combined model we will only train the generator
			self.parallel_discriminator.trainable = False
	
			# Discriminators determines validity of translated images / condition pairs
			valid = self.parallel_discriminator([fake_A, img_B])

			if True:
				self.combined = Model(inputs=[img_B], outputs=[valid,fake_A])
				self.parallel_combined = self.combined

			if True:
				self.parallel_combined.compile(loss=['binary_crossentropy', self.loss_G],
					loss_weights=[self.lambda_D, self.lambda_1],
					optimizer=self.opt_G)

		else:	# not use gan
			self.parallel_combined = self.parallel_generator

			losses_G = [self.loss_G]
			self.parallel_combined.compile(loss=losses_G,optimizer=self.opt_G,metrics=['accuracy'])

		# ======= Data ==========        
		data = load_data.load_data(nr_of_channelsA=self.channelsA,
			nr_of_channelsB=self.channelsB,
			batch_size=self.batch_size,
			subfolder=image_folder,
			nii=self.nii,
			ndim=self.ndims,
			scaling=scaling,
			final_activation=self.final_activation,
			distmap=self.calculate_distmap)
		
		self.A_train = data["trainA_images"]
		self.A_train_masks = data["trainA_masks"]
		self.A_train_valid = data["trainA_valid"]
		self.B_train = data["trainB_images"]
		self.B_train_valid = data["trainB_valid"]
		self.A_test = data["testA_images"]
		self.A_test_masks = data['testA_masks']
		self.A_test_valid = data['testA_valid']
		self.B_test = data["testB_images"]
		self.B_test_valid = data['testB_valid']
		self.testB_image_names = data["testB_image_names"]
		if self.trainmode == 'image':
			self.testA_image_z = data["testA_image_z"]
			self.testB_image_z = data["testB_image_z"]

		if initG is not None and applyFCN: 
			self.saveNiisPatch3D(epoch=0,mode='fcn')
			exit()


		if self.use_data_generator:
			if True:	

				self.data_generator = data_generator(
					self.A_train, self.A_train_masks, self.A_train_valid, 
					self.B_train, self.B_train_valid, 
					self.batch_size, self.img_shapeA,self.img_shapeB, mode="train",trainmode=self.trainmode, 
					augment_opts = self.augment_opts,
					final_activation=self.final_activation, shuffle=True,pre_norm=self.pre_norm)
				if self.use_val_data_generator:		# calculate distmap needs to be implemented
					self.val_data_generator = data_generator(
						self.A_test, self.A_test_masks, self.A_test_valid,
						self.B_test, self.B_test_valid,
						self.batch_size, self.img_shapeA,self.img_shapeB, mode="test",trainmode=self.trainmode, 
						augment_opts = None,
						final_activation=self.final_activation, shuffle=True,pre_norm=self.pre_norm)


		# ======= Create designated run folder and store meta data ==========
		directory = os.path.join('images', self.testString)
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.writeMetaDataToJSON()

		# ======= Initialize training ==========
		sys.stdout.flush()

		print("SUMMARIZE SETTINGS")

		print("DATA")
		print("image_folder: ",image_folder)
		print("ndims: ",self.ndims)
		print("img_shapeA: ",self.img_shapeA)
		print("img_shapeB: ",self.img_shapeB)

		print("PREPROCESSING")
		print("scaling: ",self.scaling)
		print("pre_norm: ",self.pre_norm)

		print("AUGMENTATION")
		print("augmentation: True")
		print("augment_brightnesscontrast: ",self.augment_opts['brightnesscontrast']) 
		print("augment_additive_gaussian_noise",self.augment_opts['additive_gaussian_noise']) 
		print("augment_gamma: ", self.augment_opts['gamma'])
		print("augment_spatial: ",self.augment_opts['spatial']) 
		print("augment_spatial_x_order", self.augment_opts['spatial_x_order'])
		print("augment_random_drop_channels",self.augment_opts['random_drop_channels'])

		print("NETWORK")
		print("network: ",self.network)
		print("nfilters: ",self.nfilters)
		print("normalization: ",str(self.normalization))
		print("final_activation: ", self.final_activation)

		print("OPTMIZER")
		print("lr_G: ",self.learning_rate_G)	
	
		print("TRAIN")
		print("trainmode: ",self.trainmode)
		print("batch_size: ",self.batch_size)
		print("workers: ", self.workers)
		print("use_gan",self.use_gan)
		
		self.train_pix2pix(epochs=self.epochs, batch_size=self.batch_size, save_interval=self.save_interval)


#===============================================================================
# Architecture functions

	def build_discriminator(self,normalization=None, strides=2,reg=None):
		
		def d_layer(layer_input, filters, f_size=4, strides=2, normalization=None, use_bias=False):
			"""Discriminator layer"""
			init = tf.random_normal_initializer(0., 0.02) #RandomNormal(stddev=0.02)	
			if self.ndims > 2:
				d = Conv3D(filters, kernel_size=f_size, strides=strides, padding='same',use_bias=use_bias, kernel_initializer=init,kernel_regularizer=reg)(layer_input)
			else:
				d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', use_bias=use_bias, kernel_initializer=init,kernel_regularizer=reg)(layer_input)
			if normalization == "batch":
				d = BatchNormalization(axis=-1)(d)
			elif normalization == "instance":
				d = InstanceNormalization(axis=-1)(d)
			d = LeakyReLU(alpha=0.2)(d)
			return d

		img_A = Input(shape=self.img_shapeA)
		img_B = Input(shape=self.img_shapeB)
		
		# Concatenate image and conditioning image by channels to produce input
		combined_imgs = Concatenate(axis=-1)([img_A, img_B])
		d1 = d_layer(combined_imgs, self.df, normalization=normalization,strides=strides, use_bias=False)   # this was WITH bias and WITHOUT normalization. 
		d2 = d_layer(d1, self.df*2,normalization=normalization,strides=strides, use_bias=False)
		d3 = d_layer(d2, self.df*4,normalization=normalization,strides=strides, use_bias=False)
		d4 = d_layer(d3, self.df*8,normalization=normalization,strides=strides, use_bias=False) 

		x = d4

		validity = Conv3D(1,kernel_size=4, strides=1, padding='same', kernel_initializer=tf.random_normal_initializer(0., 0.02))(x)

		validity = Activation('sigmoid')(validity)
		
		model = Model([img_A, img_B], validity)

		return model




#===============================================================================
# Training
	
	def train_pix2pix(self, epochs, batch_size=1, save_interval=1):
		" This function does manual training (does store images every X samples) and should be used when training GAN (pix2pix or cyclegan). 	"
		def run_training_iteration(loop_index, epoch_iterations):

			# ---------------------
			#  Train Discriminator
			# ---------------------
			
			# condition on B and generate a translated version
			fake_A = self.parallel_generator.predict(imgs_B)

			self.parallel_discriminator.trainable = True
			#print(self.discriminator.trainable) 
	
			# Train the discrimniators (original images= real / generated = Fake)
			d_loss_real = self.parallel_discriminator.train_on_batch([imgs_A, imgs_B], valid)
			d_loss_fake = self.parallel_discriminator.train_on_batch([fake_A, imgs_B], fake)
			print(self.parallel_discriminator.metrics_names)
			print(d_loss_real)
			print(d_loss_fake)
			d_loss = 0.5 * np.add(d_loss_real,d_loss_fake)

			self.parallel_discriminator.trainable = False
			#print(self.discriminator.trainable) 

			# -----------------
			#  Train Generator
			# -----------------	

			# Train the generator
			g_loss = self.parallel_combined.train_on_batch(imgs_B,[valid,imgs_A])	
			print(self.parallel_combined.metrics_names)
			print(g_loss)

			# ======= Generator training ==========
			D_losses_real.append(d_loss_real[0])
			D_losses_fake.append(d_loss_fake[0])
			D_acc_real.append(d_loss_real[1])
			D_acc_fake.append(d_loss_fake[1])
			
			D_losses.append(d_loss[0])
			G_losses.append(g_loss[0])

			print('\n')
			print('Epoch----------------', epoch, '/', epochs)
			print('Loop index----------------', loop_index + 1, '/', epoch_iterations)
			print('D_loss: ', d_loss[0])
			print('G_loss: ', g_loss[0])

			if loop_index % 20 == 0:
				fake_A = self.generator.predict(imgs_B)
				# Save temporary images continously
				if self.ndims > 2:
					sl = int(imgs_A.shape[3]/2)
					self.save_tmp_images(imgs_A[:,:,:,sl,:], imgs_B[:,:,:,sl,:], fake_A[:,:,:,sl,:])
				else:
        				self.save_tmp_images(imgs_A[:,:,:,:self.channelsA], imgs_B, fake_A)
				self.print_ETA(start_time, epoch, epoch_iterations, loop_index)


		# ======================================================================
        	# Begin training
	        # ======================================================================
		D_losses = []
		D_losses_real = []
		D_losses_fake = []
		D_acc_real = []
		D_acc_fake = []
		
		D_losses = []
		G_losses = []
		
		# # Image pools used to update the discriminators
		# not implemented
		print(self.discriminator.output_shape)

		label_shape = (batch_size,) + self.discriminator.output_shape[1:]
		print(label_shape)
		valid_dummy = np.ones(shape=label_shape) #*.9
		fake_dummy = np.zeros(shape=label_shape)

		# Start stopwatch for ETAs
		start_time = time.time()

		if self.trainmode == 'patch' and not self.use_data_generator:
			idxs = []
			for im in np.arange(len(self.A_train)):
				mask = np.ones(self.A_train[im].shape[0:-1])
				sz = self.img_shapeA[0]
				sz2 = int(sz/2)
				mask = np.pad(mask[sz2:-sz2,sz2:-sz2,sz2:-sz2],((sz2,sz2),(sz2,sz2),(sz2,sz2)),'constant')

				 # only pixels with center pixel within mask.
				mask = np.multiply(mask,np.sum(np.abs(self.A_train[im]),axis=3)>0)

				idc = np.flatnonzero(mask)
				imid = im * np.ones((len(idc),))
				ids = np.concatenate([imid[:,np.newaxis],idc[:,np.newaxis]],1)
				idxs.append(ids)
			idxs = np.concatenate(idxs,0).astype(np.int64)



		# stukje wiskidness uit https://github.com/keras-team/keras/blob/master/keras/engine/training_generator.py
		max_queue_size=50
	
		steps_per_epoch = len(self.data_generator)
		print('Steps per epoch', steps_per_epoch)
		if self.workers > 0:
			enqueuer = OrderedEnqueuer(
				self.data_generator,
				use_multiprocessing=True,
				shuffle=False)
			enqueuer.start(workers=self.workers, max_queue_size=max_queue_size)
			output_generator = enqueuer.get()	
		else:
			output_generator = iter_sequence_infinite(generator)

		for epoch in range(1, epochs + 1):
			if True:
				loop_index = 1
				steps_done = 0

				while steps_done < steps_per_epoch:
					generator_output = next(output_generator)	
					imgs_B, imgs_A = generator_output

					valid = valid_dummy #np.random.uniform(low=0.9,high=1.0,size=valid_dummy.shape)
					fake = fake_dummy
			
					run_training_iteration(loop_index, self.data_generator.__len__())
				
					# Break if loop has ended
					if loop_index >= self.data_generator.__len__():
						break

					loop_index += 1
				if self.workers == 0:
					generator.on_epoch_end()
			
			

			if epoch % self.save_interval == 0:
				self.saveModel(self.generator, epoch)

			training_history = {
				'G_losses': G_losses,
				'D_losses': D_losses,
				'D_losses_real': D_losses_real,
				'D_losses_fake': D_losses_fake,
				'D_acc_real': D_acc_real,
				'D_acc_fake': D_acc_fake,}
			self.writeLossDataToFile(training_history)


			if epoch % self.save_interval == 0:
				if self.ndims == 3 and self.trainmode == 'patch':
					self.saveNiisPatch3D(epoch,number=self.save_interval_ntest)	
				else:
					exit()

	def save_tmp_images(self, real_image_A,  real_image_B, synthetic_image_A):		
		real_image_A_hstack = []
		for idx in np.arange(real_image_A.shape[-1]):
			real_image_A_hstack.append(real_image_A[0][:,:,idx])
		real_image_A = np.hstack(real_image_A_hstack)

		synthetic_image_A_hstack = []
		for idx in np.arange(synthetic_image_A.shape[-1]):
			synthetic_image_A_hstack.append(synthetic_image_A[0][:,:,idx])
		synthetic_image_A = np.hstack(synthetic_image_A_hstack)

		real_image_B_hstack = []
		for idx in np.arange(real_image_B.shape[-1]):
			real_image_B_hstack.append(real_image_B[0][:,:,idx])
		real_image_B = np.hstack(real_image_B_hstack)

		self.truncateAndSave(None, real_image_A, synthetic_image_A, real_image_B,
			'images/{}/{}.png'.format(self.testString, 'tmp'))


	def print_ETA(self, start_time, epoch, epoch_iterations, loop_index):
		passed_time = time.time() - start_time

		iterations_so_far = ((epoch - 1) * epoch_iterations + loop_index) / self.batch_size
		iterations_total = self.epochs * epoch_iterations / self.batch_size
		iterations_left = iterations_total - iterations_so_far
		eta = round(passed_time / (iterations_so_far + 1e-5) * iterations_left)

		passed_time_string = str(datetime.timedelta(seconds=round(passed_time)))
		eta_string = str(datetime.timedelta(seconds=eta))
		print('Time passed', passed_time_string, ': ETA in', eta_string)

#===============================================================================
# Save and load

	def saveModel(self, model, epoch):
		# Create folder to save model architecture and weights
		directory = os.path.join('saved_models', self.date_time)
		if not os.path.exists(directory):
			os.makedirs(directory)

		model_path_w = 'saved_models/{}/{}_weights_epoch_{}.hdf5'.format(self.date_time, model.name, epoch)
		model.save_weights(model_path_w)
		model_path_m = 'saved_models/{}/{}_model_epoch_{}.json'.format(self.date_time, model.name, epoch)
		model.save_weights(model_path_m)
		json_string = model.to_json()
		with open(model_path_m, 'w') as outfile:
			json.dump(json_string, outfile)
		print('{} has been saved in saved_models/{}/'.format(model.name, self.date_time))


	def saveNiisPatch3D(self, epoch,mode="patch",number = 0): #fcn1
		# mode = patch or fcn
		testString = self.testString
		testDir = os.path.join('niis', testString)
		if not os.path.exists(testDir):
			os.makedirs(testDir)

		sz = self.img_shapeA[0]
		sz2 = int(sz/2)
		if number == 0:
			number = self.B_test.shape[0]
		elif number > self.B_test.shape[0]:
			number = self.B_test.shape[0]

		for img_id in np.arange(number):
			subjid = self.testB_image_names[img_id].split(".")[0]
			print(subjid)

			if mode == "patch":
				# output test images by concatenating patches.                         
				real_A = np.pad(self.A_test[img_id],((sz2,sz2),(sz2,sz2),(sz2,sz2),(0,0)),'constant')
				real_B = np.pad(self.B_test[img_id],((sz2,sz2),(sz2,sz2),(sz2,sz2),(0,0)),'constant')
				mask = np.pad(self.A_test_masks[img_id],((sz2,sz2),(sz2,sz2),(sz2,sz2)),'constant')
				
				frac = 2
				sz2frac = int(sz2/frac)			# slicing needs integers. 
				fake_A = np.zeros(real_A.shape)

				coords = []
				for x in np.arange(sz2,fake_A.shape[0] - sz2, int(sz2/frac)):
					for y in np.arange(sz2,fake_A.shape[1] - sz2 ,int(sz2/frac)):
						for z in np.arange(sz2, fake_A.shape[2] - sz2,int(sz2/frac)):
							coords.append([x,y,z])
							print([x,y,z])

				start = time.time()
				
				batch_shapeB = (len(coords),) + self.img_shapeB
				X = np.zeros(batch_shapeB)

				end = time.time()
				print(end - start)

				
				for coord_base in np.arange(0,len(coords)):

					x,y,z = coords[coord_base]
					patch_B = real_B[x-sz2:x+sz2, y-sz2:y+sz2, z-sz2:z+sz2,:]
					#patch_mask = mask[x-sz2:x+sz2, y-sz2:y+sz2, z-sz2:z+sz2]

					if self.pre_norm:
						patch_mask = mask[x-sz2:x+sz2, y-sz2:y+sz2, z-sz2:z+sz2]
						if patch_mask.sum() > 0:
							for ch in np.arange(patch_B.shape[-1]):
								tmp = patch_B[:,:,:,ch]
								sd = tmp[patch_mask>0].std()
								if sd > 0:
									patch_B[:,:,:,ch] = ((tmp - tmp[patch_mask>0].mean()) / sd) * patch_mask
								else:
									patch_B[:,:,:,ch] = ((tmp - tmp[patch_mask>0].mean())) * patch_mask

					X[coord_base] = patch_B

					end = time.time()
					print(end - start)

				end = time.time()
				print(end - start)
				print((end-start)/len(coords))

				dummy = self.parallel_generator.predict(X,batch_size=self.batch_size,verbose=1)

				end = time.time()
				print(end - start)

				for coord_base in np.arange(0,len(coords)):
					
					x,y,z = coords[coord_base]

					fake_A[x-sz2frac:x+sz2frac, y-sz2frac:y+sz2frac, z-sz2frac:z+sz2frac,:self.channelsA] = \
						dummy[coord_base,sz2-sz2frac:sz2+sz2frac,sz2-sz2frac:sz2+sz2frac,sz2-sz2frac:sz2+sz2frac,:self.channelsA]
							# dummy[0]
		
						
				real_A = real_A[sz2:-sz2,sz2:-sz2,sz2:-sz2,:self.channelsA]
				real_B = real_B[sz2:-sz2,sz2:-sz2,sz2:-sz2,:]
				fake_A = fake_A[sz2:-sz2,sz2:-sz2,sz2:-sz2,:self.channelsA]
				# end output test images by concatenating patches.  

			if epoch == self.save_interval:
				pair_img = nib.Nifti1Pair(real_A.astype(np.float32), np.eye(4))
				pair_img.header.set_data_dtype(np.float32)
				nib.save(pair_img,'{}/{}_epoch{}_realA.nii.gz'.format(
				             testDir, subjid, epoch))
				pair_img = nib.Nifti1Pair(real_B, np.eye(4))
				pair_img.header.set_data_dtype(np.float32)
				nib.save(pair_img,'{}/{}_epoch{}_realB.nii.gz'.format(
				                testDir, subjid, epoch))

			pair_img = nib.Nifti1Pair(fake_A, np.eye(4))
			pair_img.header.set_data_dtype(np.float32)
			nib.save(pair_img,'{}/{}_epoch{}_fakeA.nii.gz'.format(
			        testDir, subjid, epoch))
			
			l1_loss = np.sum(np.abs(real_A-fake_A))  #self.l1_loss(self.A_test[img_id],fake_A)

			pickle.dump(l1_loss,open('{}/{}_epoch{}_loss.p'.format(
				            testDir, subjid, epoch),"wb"))
			del real_A, fake_A, real_B



	def writeLossDataToFile(self, history):
		keys = sorted(history.keys()) 
		with open('images/{}/loss_output.csv'.format(self.testString), 'w') as csv_file:
			writer = csv.writer(csv_file, delimiter=',')
			writer.writerow(keys)
			writer.writerows(zip(*[history[key] for key in keys]))

	def writeMetaDataToJSON(self):

		directory = os.path.join('images', self.testString)
		if not os.path.exists(directory):
			os.makedirs(directory)
		# Save meta_data
		data = {}
		data['meta_data'] = []
		data['meta_data'].append({
		'ndims:': self.ndims,
		'nii:' : self.nii,
		'trainmode': self.trainmode,
		'df:' : self.df,
		'dg:' : self.gf,
		'img shapeA: height,width,channels': self.img_shapeA,
		'img_shapeB: height,width,channels': self.img_shapeB,
		'batch size': self.batch_size,
		'save interval': self.save_interval,
		'normalization function': str(self.normalization),
		'lambda_1': self.lambda_1,
		'lambda_d': self.lambda_D,
		'learning_rate_D': self.learning_rate_D,
		'learning rate G': self.learning_rate_G,
		'epochs': self.epochs,
		'use patchGan in discriminator': self.use_patchgan,
		'beta 1': self.beta_1,
		'beta 2': self.beta_2,
		'number of A train examples': len(self.A_train),
		'number of B train examples': len(self.B_train),
		'number of A test examples': len(self.A_test),
		'number of B test examples': len(self.B_test),
        })

#    def writeAugmentOptionsToJSON(self):
 #   	directory = os.path.join('augment', self.testString)
#		if not os.path.exists(directory):
#			os.makedirs(directory)
#		# Save meta_data
#		data = {}
#		data['meta_data'] = []
#		data['meta_data'].append({

#		})

#===============================================================================
# Help functions
	def truncateAndSave(self, real_, real, synthetic, reconstructed, path_name):
		print("Truncate and save")

		if len(real.shape) > 3:
			real = real[0]
			synthetic = synthetic[0]
			reconstructed = reconstructed[0]

		# Append and save
		if real_ is not None:
			if len(real_.shape) > 4:
				real_ = real_[0]
			image = np.hstack((real_[0], real, synthetic, reconstructed))
		else:
			image = np.hstack((real, synthetic, reconstructed))

	        #if self.channels == 1:
        	        #image = image[:, :, 0]
		if self.scaling == "mask_min-max":
			#toimage(image, cmin=-1, cmax=1).save(path_name)
			Image.fromarray(image).convert("L").save(path_name)
		elif self.scaling == "mask_variance" :
			#toimage(image, cmin=-1, cmax=1).save(path_name)
			Image.fromarray(image).convert("L").save(path_name)
		else:
			#toimage(image, cmin=0, cmax=1).save(path_name)
			Image.fromarray(image).convert("L").save(path_name)



if __name__ == '__main__':

	parser = optparse.OptionParser()
	parser.add_option('','--image_folder',action="store", type="string", dest="image_folder",help="Folder to use",default="")

	parser.add_option('','--ndims', action="store", type="int", dest="ndims",help="Number of dimensions",default=2)

	parser.add_option('','--shapeAi', action="store", type="int", dest="shapeAi",help="Patch i size image A",default=256)
	parser.add_option('','--shapeAj', action="store", type="int", dest="shapeAj",help="Patch j size image A",default=256)
	parser.add_option('','--shapeAk', action="store", type="int", dest="shapeAk",help="Patch k size image A",default=256)
	parser.add_option('','--channelsA', action="store", type="int", dest="channelsA",help="Number of channels image A",default=1)

	parser.add_option('','--shapeBi', action="store", type="int", dest="shapeBi",help="Patch i size image B",default=256)
	parser.add_option('','--shapeBj', action="store", type="int", dest="shapeBj",help="Patch j size image B",default=256)
	parser.add_option('','--shapeBk', action="store", type="int", dest="shapeBk",help="Patch k size image B",default=256)
	parser.add_option('','--channelsB', action="store", type="int", dest="channelsB",help="Number of channels image B",default=1)

	parser.add_option('','--scaling', action="store", type="string",dest="scaling", help="Scale continuous data <variance,min-max,mask_variance,mask_min-max> (default: mask_variance)", default="mask_variance")
	parser.add_option('','--pre_norm',action="store_true",dest="pre_norm",help="Use pre_normalization of train and test samples (variance scaling of each independent sample) (default: False) ")

	parser.add_option('','--network',action="store", type="string", dest="network", help="Network to use <ae,isenseeX-X,unet> (default: unet)", default="unet")
	parser.add_option('','--nfilters',action="store", type="int", dest="nfilters",help="Base number of filters to use (default: 64)", default=64)
	parser.add_option('','--normalization',action="store", type="string",dest="normalization", help="Normalization to use <none, batch, instance> (default: none)", default=None)
	parser.add_option('','--final_activation',action="store", type="string", dest="final_activation", help="Final activation of generator <tanh, sigmoid, softmax, linear> (default: linear)", default="tanh")
	
	parser.add_option('','--lr_G', action="store", type="float", dest="lr_G", help="Learning rate (default: 0.001)", default=0.001)
	parser.add_option('','--regularizer',action="store", type="string", dest="regularizer", help="Regularizer: None, l1(0.0005), l2(0.0005) (default: l2)", default="l2")
	parser.add_option('','--loss_G',action="store", type="string", dest="loss_G", help="mae <default> or mse", default="mae")
	parser.add_option('','--batch_size',action="store", type="int", dest="batch_size", help="Batch size to use (default: 1)", default=1)
	parser.add_option('','--trainmode',action="store", type="string", dest="trainmode", help="image or patch", default="")

	
	parser.add_option('','--augment_brightnesscontrast', action="store", type="string",dest="augment_brightnesscontrast", help="Use brightness/contrast augmentation (default: False)",default=False)
	parser.add_option('','--augment_gamma', action="store_true", dest="augment_gamma", help="Use gamma augmentation (default: True)",default=True)

	parser.add_option('','--augment_additive_gaussian_noise', action="store", type="string",dest="augment_additive_gaussian_noise", help="Use additive gaussian noise augmentation (default: False)",default=False)

	parser.add_option('','--augment_spatial',action="store_true", dest="augment_spatial",help="Use spatial augmentation (default: False)",default=False)
	parser.add_option('','--augment_spatial_x_order',action="store", type="int", dest="augment_spatial_x_order", help="Order for interpolation of spatial augmentation (lower values increase preprocessing speed)", default=3)

	parser.add_option('','--augment_random_drop_channels', action="store", type="string",dest="augment_random_drop_channels", help="Randomly drop channels (default: False)",default=False)
	parser.add_option('','--augment_variable_blurring', action="store", type="string",dest="augment_variable_blurring", help="Random variable blurring (default: False)",default=False)
	
	parser.add_option('','--use_gan', action="store_true", dest="use_gan", help="Use GAN", default=False)
	parser.add_option('','--workers', action="store",type="int", dest="workers", help="Number of workers (default=1)", default=4)

	parser.add_option('','--save_interval', action="store",type="int", dest="save_interval", help="Interval to save test results", default=5)
	parser.add_option('','--save_interval_ntest', action="store", type="int", dest="save_interval_ntest", help="Save n test cases each interval, 0 = all test cases (default=0)", default=0)


	parser.add_option('','--applyFCN', action="store_true", dest="applyFCN",help="Apply generator weights through FCN", default=False)
	parser.add_option('','--initG', action="store",type="string",dest="initG", help="Weights to initialize Generator", default=None)

	options, args = parser.parse_args()

	image_shapeA = (options.shapeAi, options.shapeAj,options.shapeAk,options.channelsA)
	image_shapeB = (options.shapeBi, options.shapeBj,options.shapeBk,options.channelsB)

	augment_opts = {'brightnesscontrast':options.augment_brightnesscontrast,
					'gamma':options.augment_gamma,
					'additive_gaussian_noise': options.augment_additive_gaussian_noise,
					'spatial': options.augment_spatial,
					'spatial_x_order': options.augment_spatial_x_order,
					'random_drop_channels': options.augment_random_drop_channels,
					'variable_blurring': options.augment_variable_blurring }


	GAN = DeepContrast(image_folder = options.image_folder,
		image_shapeA=image_shapeA, image_shapeB=image_shapeB,

		network=options.network, 
		nfilters=options.nfilters,
		normalization=options.normalization,
		final_activation=options.final_activation,

		pre_norm=options.pre_norm,

		lr_G=options.lr_G,
		loss_G=options.loss_G, 
		batch_size=options.batch_size, 
		trainmode=options.trainmode, 
		date_time_string_addition='_pix2pix',

		augment_opts=augment_opts,
		
		use_gan=options.use_gan,
		workers=options.workers,
		save_interval=options.save_interval,
		save_interval_ntest=options.save_interval_ntest,

		applyFCN=options.applyFCN,
		initG=options.initG
		)
