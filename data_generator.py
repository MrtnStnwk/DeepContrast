
from tensorflow.python import keras
import numpy as np
import numpy.ma as ma
import random
from scipy.ndimage import distance_transform_edt as distance
from augment import augment_3d
from scipy import ndimage
import matplotlib.pyplot as plt

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html
# https://github.com/keras-team/keras/issues/8130

class data_generator(keras.utils.data_utils.Sequence):
	'Generates data for Keras'

	def __init__(self, A, A_masks, A_valid, B, B_valid, batch_size, image_shapeA, image_shapeB, mode, trainmode, augment_opts, 
		final_activation, shuffle, pre_norm = True):
		
		self.ndims = len(image_shapeA)-1

		self.A = A
		print(self.A.shape)

		self.A_masks = A_masks
		self.A_valid = A_valid
		self.B = B
		self.B_valid = B_valid
		self.batch_size = batch_size	
		self.image_shapeA = image_shapeA
		self.image_shapeB = image_shapeB
		self.mode = mode
		self.trainmode = trainmode
		self.augment_opts = augment_opts
		self.final_activation = final_activation
		self.shuffle = shuffle
		self.pre_norm = pre_norm
		 
		if self.trainmode == 'patch':
			idxs = []

			sz = self.image_shapeA[0]
			sz2 = int(sz/2)
			base_mask = np.ones(self.A[0].shape[0:-1])

			base_mask = np.pad(base_mask[sz2:-sz2,sz2:-sz2,sz2:-sz2],((sz2,sz2),(sz2,sz2),(sz2,sz2)),'constant')

			for im in np.arange(len(self.A)):
				mask = self.A_masks[im] +1.
				idc = np.flatnonzero(mask)
				imid = im * np.ones((len(idc),))
				ids = np.concatenate([imid[:,np.newaxis],idc[:,np.newaxis]],1)
				idxs.append(ids)
			self.list_IDs = np.concatenate(idxs,0).astype(np.int)
		else:
			exit()

		self.on_epoch_end()
		

	def __len__(self):
		'Denotes the number of batches per epoch'
		if self.trainmode == 'patch':
			n = np.sum(self.A_masks,axis=(0,1,2,3)) / (self.image_shapeA[0] * self.image_shapeA[1] * self.image_shapeA[2])  *  8 # factor 8 oversampling 
		elif self.trainmode == 'image':
			n =  len(self.list_IDs) 
		else:
			exit()

		return int(np.floor(n)/self.batch_size)


	def __getitem__(self,index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		
		# Find array of IDs
		list_IDs_tmp = [self.list_IDs[k,] for k in indexes]

		# Generate data
		if self.trainmode == 'patch':
			X, y = self.__data_generation_patch(list_IDs_tmp)
		
		return X,y


	def on_epoch_end(self):
		print('Updates indexes after each epoch')

		self.indexes = np.arange(self.list_IDs.shape[0])
		if self.shuffle == True:
			np.random.shuffle(self.indexes)


	def get_patch3d(self,x, ix, iy, iz, sz):
		sz2 = int(sz/2)
		if len(x.shape) > 3:
			patch = np.zeros((sz,sz,sz,x.shape[3]),dtype=x.dtype)
		else:	
			patch = np.zeros((sz,sz,sz),dtype=x.dtype)

		min_allowable_offset_x = -ix
		max_allowable_offset_x = x.shape[0] - ix 

		min_allowable_offset_y = -iy
		max_allowable_offset_y = x.shape[1] - iy

		min_allowable_offset_z = -iz
		max_allowable_offset_z = x.shape[2] - iz

		neg_offset_x = max(min_allowable_offset_x, -sz2)
		pos_offset_x = min(max_allowable_offset_x, sz2)

		neg_offset_y = max(min_allowable_offset_y, -sz2)
		pos_offset_y = min(max_allowable_offset_y, sz2)

		neg_offset_z = max(min_allowable_offset_z, -sz2)
		pos_offset_z = min(max_allowable_offset_z, sz2)

		patch[sz2+neg_offset_x: sz2+pos_offset_x,
			sz2+neg_offset_y: sz2+pos_offset_y,
			sz2+neg_offset_z: sz2+pos_offset_z] = x[ix+neg_offset_x: ix+pos_offset_x,
													iy+neg_offset_y: iy+pos_offset_y,
													iz+neg_offset_z: iz+pos_offset_z]
		return patch


	def  __data_generation_patch(self, IDs):
		'Generates data containing batch_size samples' 

		# Initialization
		X = np.empty((self.batch_size,) + self.image_shapeB)
		y = np.empty((self.batch_size,) + self.image_shapeA)
		sz = self.image_shapeA[0]
		sz2 = int(sz/2)

		# Generate data
		for i, ID in enumerate(IDs):

			ix, iy, iz  = np.unravel_index(ID[1], self.A[ID[0]].shape[0:-1])


			A_valid, B_valid = None, None
			if self.A_valid is not None:
				A_valid = np.copy(self.B_valid[ID[0]])
			if self.B_valid is not None:
				B_valid = np.copy(self.B_valid[ID[0]])

			if self.augment_opts is not None and self.ndims == 3 and not self.augment_opts['spatial']:
				patch_A = self.get_patch3d(self.A[ID[0]],ix,iy,iz,sz)
				patch_B = self.get_patch3d(self.B[ID[0]],ix,iy,iz,sz)
				mask = self.get_patch3d(self.A_masks[ID[0]],ix,iy,iz,sz)

				patch_A,patch_B, mask = augment_3d(patch_A, patch_B,  
					augment_opts=self.augment_opts,
					final_activation=self.final_activation, 
					mask=mask,
					A_valid=A_valid,
					B_valid=B_valid)
			elif self.augment_opts is not None and self.ndims == 3 and self.augment_opts['spatial']:
				offset = int(sz/4)

				patch_A = self.get_patch3d(self.A[ID[0]],ix,iy,iz,sz + 2 * offset)
				patch_B = self.get_patch3d(self.B[ID[0]],ix,iy,iz,sz + 2 * offset)
				mask = self.get_patch3d(self.A_masks[ID[0]],ix,iy,iz,sz + 2 * offset)

				ty, tX, tmask = augment_3d(patch_A, patch_B,  
					augment_opts=self.augment_opts,
					final_activation=self.final_activation, 
					mask=mask,
					A_valid=A_valid,
					B_valid=B_valid)

				patch_A = ty[offset:-offset,offset:-offset,offset:-offset,:]
				patch_B = tX[offset:-offset,offset:-offset,offset:-offset,:]
				mask = tmask[offset:-offset,offset:-offset,offset:-offset]
			
			else:
				patch_A = np.copy(self.A[ID[0],ix-sz2:ix+sz2,iy-sz2:iy+sz2,iz-sz2:iz+sz2,:])
				patch_B = np.copy(self.B[ID[0],ix-sz2:ix+sz2,iy-sz2:iy+sz2,iz-sz2:iz+sz2,:])
				mask = np.copy(self.A_masks[ID[0],ix-sz2:ix+sz2,iy-sz2:iy+sz2,iz-sz2:iz+sz2])

			
			if self.pre_norm and mask.sum() > 0:
				for ch in np.arange(patch_B.shape[-1]):
					tmp = patch_B[:,:,:,ch]
					sd = tmp[mask>0].std()
					if sd > 0:
						patch_B[:,:,:,ch] = ((tmp - tmp[mask>0].mean()) / sd) * mask
					else:
						patch_B[:,:,:,ch] = ((tmp - tmp[mask>0].mean())) * mask
			
			X[i,] = patch_B
			y[i,] = patch_A

		return X,y
