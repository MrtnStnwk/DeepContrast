
import numpy as np
import random
import scipy
from matplotlib import pyplot as plt


# use random.random nstead of np.random.random. Otherwise the same numbers will be produced in multiworker state. see third post here: https://github.com/numpy/numpy/issues/9650

def transform_matrix_offset_center_3d(matrix, x, y, z):
	o_x = float(x) / 2 + 0.5
	o_y = float(y) / 2 + 0.5
	o_z = float(z) / 2 + 0.5
	offset_matrix = np.array([[1, 0, 0, o_x], [0, 1, 0, o_y], [0, 0, 1, o_z], [0, 0, 0, 1]])
	reset_matrix = np.array([[1, 0, 0, -o_x], [0, 1, 0, -o_y], [0, 0, 1, -o_z], [0, 0, 0, 1]])
	transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
	return transform_matrix

def apply_affine_transform_3d(x, theta_x=0, theta_y=0, theta_z=0, tx=0, ty=0, tz=0, zx=1, zy=1, zz=1,
				row_axis=0, col_axis=1, depth_axis = 2, channel_axis=3,
				fill_mode='nearest', cval=0., order=1):
	"""Applies an affine transformation specified by the parameters given.
		x: 2D numpy array, single image
		theta: rotation angle in degrees
		tx: width shift
		ty: heigh shift
		channel_axis: Index of channel axis (default 2)
		order: int, order of interpolation (default: 3)
	# Returns
		The transformed version of the input
	"""
	
	transform_matrix = None

	if theta_z != 0:
		theta_z = np.deg2rad(theta_z)
		rotation_matrix = np.array([[np.cos(theta_z), -np.sin(theta_z), 0, 0],
									[np.sin(theta_z), np.cos(theta_z), 0, 0],
									[0, 0, 1, 0],
									[0, 0, 0,1]])
		transform_matrix = rotation_matrix

	if theta_x != 0:
		theta_x = np.deg2rad(theta_x)
		rotation_matrix = np.array([[1, 0, 0, 0],
									[0, np.cos(theta_x), -np.sin(theta_x),  0],
									[0, np.sin(theta_x), np.cos(theta_x), 0],
									[0, 0, 0,1]])
		if transform_matrix is None:
			transform_matrix = rotation_matrix
		else:
			transform_matrix = np.dot(transform_matrix, rotation_matrix)

	if theta_y != 0:
		theta_y = np.deg2rad(theta_y)
		rotation_matrix = np.array([[np.cos(theta_y), 0, -np.sin(theta_y), 0],
									[0, 1, 0,  0],
									[np.sin(theta_y),0, np.cos(theta_y), 0],
									[0, 0, 0, 1]])
		if transform_matrix is None:
			transform_matrix = rotation_matrix
		else:
			transform_matrix = np.dot(transform_matrix, rotation_matrix)

	if tx != 0 or ty != 0 or tz != 0:
		shift_matrix = np.array([[1, 0, 0, tx],
					[0, 1, 0, ty],
					[0, 0, 1, tz],
					[0, 0, 0, 1]])
		if transform_matrix is None:
			transform_matrix = shift_matrix
		else:
			transform_matrix = np.dot(transform_matrix, shift_matrix)

	if zx != 1 or zy != 1 or zz != 1:
		zoom_matrix = np.array([[zx, 0, 0, 0],
			[0, zy, 0, 0],
			[0, 0, zz, 0],
            [0, 0, 0, 1]])
		if transform_matrix is None:
			transform_matrix = zoom_matrix
		else:
			transform_matrix = np.dot(transform_matrix, zoom_matrix)


	if transform_matrix is not None:
		h, w, d  = x.shape[row_axis], x.shape[col_axis], x.shape[depth_axis]
		transform_matrix = transform_matrix_offset_center_3d(
				transform_matrix, h, w, d)
		x = np.rollaxis(x, channel_axis, 0)
		final_affine_matrix = transform_matrix[:3, :3]
		final_offset = transform_matrix[:3, 3]

		channel_images = [scipy.ndimage.interpolation.affine_transform(
			x_channel,
			final_affine_matrix,
			final_offset,
			order=order,
			mode=fill_mode,
			cval=cval) for x_channel in x]
		x = np.stack(channel_images, axis=0)
		x = np.rollaxis(x, 0, channel_axis + 1)
	return x

def augment_3d(y, x , augment_opts=None, 
	final_activation=None, mask=None, A_valid=None, B_valid=None):
	def augment_channel_3d(x,final_activation):
		def adjust_brightness(x,delta):
			return x + delta
		def adjust_contrast(x, contrast_factor):
			return x * contrast_factor
		def random_brightness(x, max_delta):
			delta = random.uniform(-max_delta, max_delta)
			return adjust_brightness(x,delta)
		def random_contrast(x, lower, upper):
			contrast_factor = random.uniform(lower, upper)
			return adjust_contrast(x,contrast_factor)

		x = random_contrast(x, 0.3, 1.3)
		x = random_brightness(x, 0.3)	

		if final_activation == "tanh":
			x = np.clip(x,-1.0,1.0)
		return x


	if random.randint(0,1) > 0:
		x = np.flip(x,0)
		if y is not None:
			y = np.flip(y,0)
		if mask is not None:
			mask = np.flip(mask,0)	
	if random.randint(0,1) > 0:
		x = np.flip(x,1)
		if y is not None:
			y = np.flip(y,1)
		if mask is not None:
			mask = np.flip(mask,1)
	if random.randint(0,1) > 0:
		x = np.flip(x,2)
		if y is not None:
			y = np.flip(y,2)
		if mask is not None:
			mask = np.flip(mask,2)

	if random.uniform(0,1) < 0.50 and augment_opts['spatial']:
		max_angle = 90
		angle_x = random.uniform(-max_angle, max_angle)
		angle_y = random.uniform(-max_angle, max_angle)
		angle_z = random.uniform(-max_angle, max_angle)

		max_disp = 0#(x.shape[0] / 1.5) / 2 			# omdat we in spatial norm zijn is de vanuit data_generator aangeboden patch al 1.5 keer groter dan nodig. Daardoor delen en dan gedeelddoor 2 is maximale verplaatsing. 

		shift_x = random.uniform(-max_disp,max_disp)
		shift_y = random.uniform(-max_disp,max_disp)
		shift_z = random.uniform(-max_disp,max_disp)

		zx = random.uniform(0.8,1.2)
		zy = random.uniform(0.8,1.2)
		zz = random.uniform(0.8,1.2)

		x = apply_affine_transform_3d(x, theta_x=-angle_x, theta_y=-angle_y,theta_z=-angle_z, tx=shift_x, ty=shift_y, tz=shift_z, zx=zx, zy=zy, zz=zz, order=augment_opts['spatial_x_order'],fill_mode='constant',cval=0.)
		y = apply_affine_transform_3d(y, theta_x=-angle_x, theta_y=-angle_y,theta_z=-angle_z, tx=shift_x, ty=shift_y, tz=shift_z, zx=zx, zy=zy, zz=zz, order=3,fill_mode='constant',cval=0.)
		if mask is not None:
		 	mask = apply_affine_transform_3d(mask[:,:,:,np.newaxis], theta_x=-angle_x, theta_y=-angle_y,theta_z=-angle_z, tx=shift_x, ty=shift_y, tz=shift_z, zx=zx, zy=zy, zz=zz, order=0,fill_mode='constant',cval=0.)[:,:,:,0]
	
	# additive gaussian noise in 30% of the patches
	if augment_opts['additive_gaussian_noise'] and random.uniform(0,1) < 0.30:
        	x += np.random.normal(0,0.2/4,x.shape)

	# variable blurring in 30% of the patches
	if augment_opts['variable_blurring'] and random.uniform(0,1) < 0.30:
		sigma = random.uniform(0.2,1.5)
		#print sigma
		for ch in np.arange(x.shape[-1]):
			x[:,:,:,ch] = scipy.ndimage.filters.gaussian_filter(x[:,:,:,ch],sigma)


	# variable gamma augmentation in 50% of the patches. 
	if augment_opts['gamma'] and random.uniform(0,1) < .50:
		for ch in np.arange(x.shape[-1]):
			gamma = random.uniform(0.8,1.5)

			tmp = x[:,:,:,ch]
			tmp_min = np.amin(tmp)
			tmp_max = np.amax(tmp)

			if (tmp_max - tmp_min) > 0 :

				# normalize to 0,1
				tmp = (tmp - tmp_min) / (tmp_max-tmp_min)
				# raise to the power
				tmp = np.power(tmp, gamma)
				# normalize to original range
				tmp = (tmp * (tmp_max-tmp_min)) + tmp_min 

				x[:,:,:,ch] = tmp


	# randomly drop a number of contrast in 30% of the cases(2 need to remain)
	if augment_opts['random_drop_channels'] and random.uniform(0,1) < 0.0: #0.3:
		ids = np.arange(x.shape[-1])
		new_order = [a for a in (ids[B_valid > 0]) ]

		random.shuffle(new_order)

		if len(new_order) > 1:
			n = random.randint(1,len(new_order))
			new_order2 = new_order[:n]

			for ch in np.arange(x.shape[-1]):
				if ch in new_order2:
					pass
				else:

			#		print('wipe channel ', ch)
					x[:,:,:,ch] *= 0.0 
		else:
			#print('insufficient number of valid channels')
			pass 

	if augment_opts['brightnesscontrast']:
		for ch in np.arange(x.shape[-1]):
			if mask is not None:
				x[:,:,:,ch] = np.multiply(augment_channel_3d(x[:,:,:,ch],final_activation), mask)
			else:
				x[:,:,:,ch] = augment_channel_3d(x[:,:,:,ch],final_activation)

	if mask is not None:
		for ch in np.arange(y.shape[-1]):
			y[:,:,:,ch] = np.multiply(y[:,:,:,ch],mask)

	return (y,x,mask)
