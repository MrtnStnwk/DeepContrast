import sys
sys.path.insert(0,"/local/m.steenwijk/intensity-normalization/")
#from intensity_normalization.normalize import kde

import os
from tempfile import mkdtemp
import numpy as np
from PIL import Image
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.np_utils import to_categorical
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as distance
#from skimage.io import imread

import matplotlib.pyplot as plt
import nibabel as nib
import warnings

def load_data(nr_of_channelsA, nr_of_channelsB, batch_size=1, 
	      nr_A_train_imgs=None, nr_B_train_imgs=None,
              nr_A_test_imgs=None, nr_B_test_imgs=None, subfolder='',
              generator=False, D_model=None, use_multiscale_discriminator=False, use_supervised_learning=False, REAL_LABEL=1.0, nii=False, ndim = 2, scaling="mask_variance",
              final_activation="linear",distmap=False):

	trainA_path_array = []
	testA_path_array = []
	trainA_image_names_array = [];
	testA_image_names_array = [];

	train_image_names = [];
	test_image_names = [];

	for idx in np.arange(nr_of_channelsA):
		trainA_path = os.path.join('data', subfolder, 'trainA'+str(idx))
		testA_path = os.path.join('data', subfolder, 'testA'+str(idx))

		trainA_image_names = sorted(os.listdir(trainA_path))

		if nr_A_train_imgs != None:
			trainA_image_names = trainA_image_names[:nr_A_train_imgs]

		testA_image_names = sorted(os.listdir(testA_path))

		if nr_A_test_imgs != None:
			testA_image_names = testA_image_names[:nr_A_test_imgs]

		trainA_path_array.append(trainA_path)
		testA_path_array.append(testA_path)
		trainA_image_names_array.append(trainA_image_names)
		testA_image_names_array.append(testA_image_names)

	train_image_names_array = sorted(os.listdir(trainA_path_array[0]))
	test_image_names_array = sorted(os.listdir(testA_path_array[0]))
	print(train_image_names)

	trainB_path_array = []
	testB_path_array = []
	trainB_image_names_array = [];
	testB_image_names_array = [];
	for idx in np.arange(nr_of_channelsB):
		trainB_path = os.path.join('data', subfolder, 'trainB'+str(idx))
		testB_path = os.path.join('data', subfolder, 'testB'+str(idx))

		# we expect the same images to be present as in trainA. 
		trainB_image_names = sorted(os.listdir(trainB_path))

		if nr_B_train_imgs != None:
			trainB_image_names = trainB_image_names[:nr_B_train_imgs]

		testB_image_names = sorted(os.listdir(testB_path))

		if nr_B_test_imgs != None:
			testB_image_names = testB_image_names[:nr_B_test_imgs]

		trainB_path_array.append(trainB_path)
		testB_path_array.append(testB_path)
		trainB_image_names_array.append(trainB_image_names)
		testB_image_names_array.append(testB_image_names)


	pid = str(os.getpid())


	if scaling == "mask_kde":
		scalingA = "None"
		scalingB = "mask_variance"
	else:
		scalingA = scaling
		scalingB = scaling


	if nii and ndim == 3:

		# print(trainA_image_names_array)
		# print(trainA_path_array)

		# print(trainB_image_names_array)
		# print(trainB_path_array)
		# exit()

		trainA_images, trainA_masks, trainA_valid = create_image_array_nii_3d(train_image_names_array, trainA_path_array, nr_of_channelsA, mmap_prefix=pid+"_trainA",scaling=scaling,final_activation=final_activation, distmap=distmap)


		if len(testA_image_names_array[0]) > 0:
			testA_images, testA_masks, testA_valid = create_image_array_nii_3d(test_image_names_array, testA_path_array, nr_of_channelsA,mmap_prefix=pid+"_testA",scaling=scaling,final_activation=final_activation, distmap=distmap)
		else:
			testA_images, testA_masks, testA_valid = [], [], []

		if nr_of_channelsB > 0:
			trainB_images,trainB_masks, trainB_valid = create_image_array_nii_3d(train_image_names_array, trainB_path_array, nr_of_channelsB,mmap_prefix=pid+"_trainB",scaling=scaling, final_activation=final_activation )
			testB_images, testB_masks, testB_valid = create_image_array_nii_3d(test_image_names_array, testB_path_array, nr_of_channelsB,mmap_prefix=pid+"_testB",scaling=scaling,final_activation=final_activation)
		else:
			trainB_images, trainB_masks,trainB_valid = [], [], []
			testB_images, testB_masks, trainB_valid = [], [], []

		if not 'test_image_names_array' in locals():
			test_image_names_array = []

		return {"trainA_images": trainA_images,"trainA_masks": trainA_masks,"trainA_valid": trainA_valid,
				"trainB_images": trainB_images,"trainB_masks": trainB_masks, "trainB_valid": trainB_valid,
				"testA_images": testA_images, "testA_masks": testA_masks,"testA_valid": testA_valid,
				"testB_images": testB_images, "testB_masks": testB_masks,"testB_valid": testB_valid,
				"testB_image_names": test_image_names_array}
	else:
		exit()											


def create_image_array_nii_3d(image_list_array, image_path_array, nr_of_channels,mmap_prefix,scaling="mask_variance", final_activation="linear", distmap=False):
	#nsubj=len(image_list_array[0])
	nsubj=len(image_list_array)

	nchann=nr_of_channels

	filename = os.path.join('', mmap_prefix+'_mmap.dat')
	filename_mask = os.path.join('', mmap_prefix+'_mask_mmap.dat')
	print('Mmap cache: ',filename)


	
	w, h, d = 182,218,182
	w, h, d = 240,240,156
	w, h, d = 182,218,156
	
	fp = np.memmap(filename, dtype='float32', mode='w+', shape=(nsubj,w,h,d,nchann))
	fpm = np.memmap(filename_mask, dtype='bool', mode='w+', shape=(nsubj,w,h,d))

	valid = np.zeros((nsubj, nchann),dtype=np.bool)

	for subjidx in np.arange(nsubj):
		for channidx in np.arange(nchann):
			#try:
				#image_nii = nib.load(os.path.join(os.path.join(image_path_array[channidx], image_list_array[channidx][subjidx])))
				image_nii = nib.load(os.path.join(os.path.join(image_path_array[channidx], image_list_array[subjidx])))
				#mask_nii = nib.load(os.path.join(os.path.join(image_path_array[channidx]+'_mask', image_list_array[channidx][subjidx])))
				mask_nii = nib.load(os.path.join(os.path.join(image_path_array[channidx]+'_mask', image_list_array[subjidx])))

				print(image_list_array[subjidx],sep=" ", end=" ", flush=True)
				image3d = image_nii.get_data().astype(np.float32)
				mask3d = mask_nii.get_data()

				image3d_pad_x = int((w - image3d.shape[0]) / 2) #0 #int((256 - image3d.shape[0]) / 2)
				image3d_pad_y = int((h - image3d.shape[1]) / 2) #0 #int((256 - image3d.shape[1]) / 2)
				image3d_pad_z = int((d - image3d.shape[2]) / 2) #0 #int((256 - image3d.shape[2]) / 2)


				prct98 = np.percentile(image3d,100)
				mu = np.mean(image3d)
				std = np.std(image3d)
				mask_max = np.percentile(image3d[mask3d>0],100)
				mask_mu = np.mean(image3d[mask3d>0])
				mask_std = np.std(image3d[mask3d>0])

				if image3d_pad_x > 0 or image3d_pad_y > 0 or image3d_pad_z > 0:
					image3d = np.pad(image3d,((image3d_pad_x,image3d_pad_x),(image3d_pad_y,image3d_pad_y),(image3d_pad_z,image3d_pad_z)),'constant')
					mask3d =  np.pad(mask3d,((image3d_pad_x,image3d_pad_x),(image3d_pad_y,image3d_pad_y),(image3d_pad_z,image3d_pad_z)),'constant')
				else:
					image3d = image3d[-image3d_pad_x:image3d_pad_x-1,-image3d_pad_y:image3d_pad_y-1,-image3d_pad_z:image3d_pad_z-1]
					mask3d = mask3d[-image3d_pad_x:image3d_pad_x-1,-image3d_pad_y:image3d_pad_y-1,-image3d_pad_z:image3d_pad_z-1]
				
				if scaling == "min-max":
					pass 
				
				elif scaling == "mask_min-max":
					image3d = np.multiply(image3d / mask_max,mask3d)
					image3d = np.clip(image3d * 2 - 1, -1.0, 1.0)
				elif scaling == "variance":
					image3d = (image3d - mu) / std
				elif scaling == "mask_variance":
					if mask_std > 0:
						image3d = np.multiply((image3d - mask_mu) / mask_std,mask3d)		
					image3d = np.clip(image3d / 4, -1.0, 1.0)
				elif scaling == "None":
					image3d = np.clip(image3d , -1.0,1.0)
				else:
					print("No valid scaling")
					exit()
				fp[subjidx,:image3d.shape[0],:image3d.shape[1],:image3d.shape[2],channidx] = image3d
				fpm[subjidx,:image3d.shape[0],:image3d.shape[1],:image3d.shape[2]] = (mask3d > 0)

				valid[subjidx,channidx] = True
			#except:
			#	warnings.warn(os.path.join(os.path.join(image_path_array[channidx], image_list_array[subjidx])) + 'not found, setting this volume to zeros')
			#	valid[subjidx,channidx] = False

	return fp, fpm, valid 
	

if __name__ == '__main__':
    load_data()
