import tensorflow as tf
tf.compat.v1.disable_eager_execution()			# -> for multi_gpu_model to work
from tensorflow.python.keras.initializers import RandomNormal, Orthogonal
#from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.python.keras.layers import Add, Activation, BatchNormalization, Concatenate, Conv2D, Conv3D, Dropout, Input, Multiply, SpatialDropout2D, SpatialDropout3D, UpSampling2D, UpSampling3D, Conv2DTranspose, Conv3DTranspose, Lambda, AveragePooling2D, AveragePooling3D, MaxPooling2D, MaxPooling3D,GlobalAveragePooling2D, Dense,Reshape, GlobalAveragePooling3D, Flatten
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU, PReLU
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow_addons.layers import InstanceNormalization

 
def build_generator_isensee2017_3d(X_shape, n_base_filters, n_labels, activation_name, depth=6, n_segmentation_levels=1, multi_input=0,  
	 use_attention=False, dropout_rate=0.3, normalization=None, reg=None):
	# Isensee et al 2017; Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge
	print('Final activation', activation_name)
	conv_activation = LeakyReLU(alpha=1e-2)

	def create_convolution_block(input_layer, n_filters, normalization = None, kernel=(3,3,3), activation=None, padding='same', strides=(1,1,1), use_bias=False):
		init = tf.random_normal_initializer(0., 0.02) #RandomNormal(stddev=0.02)

		#print('n_filters', n_filters)
		layer = Conv3D(n_filters, kernel, padding=padding, strides=strides,use_bias=use_bias, kernel_initializer=init,kernel_regularizer=reg)(input_layer)
		if normalization == "batch":
			#print('Use batch normalization')
			layer = BatchNormalization(axis=-1)(layer)
		elif normalization == "instance":
			#print('Use instance normalization')
			layer = InstanceNormalization(axis=-1)(layer)

		if activation is None:
			return layer
		elif activation == 'relu':
			return Activation('relu')(layer)
		elif activation == 'sigmoid':
			return Activation('sigmoid')(layer)
		elif activation == 'leaky':
			return LeakyReLU(alpha=1e-2)(layer)
		else:
			return activation(layer)

	def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_last"):
		convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters,activation=conv_activation, normalization=normalization)
		dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
		convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters,activation=conv_activation, normalization=normalization)
		return convolution2

	def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
		up_sample = UpSampling3D(size=size)(input_layer)
		convolution = create_convolution_block(up_sample, n_filters, activation=conv_activation, normalization=normalization)
		return convolution

	def create_localization_module(input_layer, n_filters):
		convolution1 = create_convolution_block(input_layer, n_filters, activation=conv_activation, normalization=normalization)			 # this *2 was not present in orginal code
		
		if True:			# resnet in decoder. 
			convolution2 = create_convolution_block(convolution1, n_filters, kernel=(3, 3, 3), activation=conv_activation, normalization=normalization)
			convolution3 = create_convolution_block(convolution2, n_filters, kernel=(3, 3, 3), activation=conv_activation, normalization=normalization)
			summation_layer = Add()([convolution1, convolution3])
			out = summation_layer
		else:
			convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1), activation=conv_activation, normalization=normalization)		# original implementeation was (1,1)
			out = convolution2
		return out

	def create_attention_block(x,g, n_filters):
		def expend_as(tensor, rep):
			#my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep},  name='psi_up'+name)(tensor)
			my_repeat = Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=4), arguments={'repnum': rep})(tensor)
			return my_repeat

		shape_x = K.int_shape(x)  # 32
		x1 = create_convolution_block(x, n_filters, kernel=(2,2,2),strides=(2,2,2),activation=None, normalization=normalization)
		g1 = create_convolution_block(g, n_filters, kernel=(1,1,1),activation=None, normalization=normalization)
		
		g1_x1 = Add()([g1,x1])
		psi = Activation('relu')(g1_x1)
		psi = create_convolution_block(psi,1,kernel=(1,1,1),activation='sigmoid',normalization=normalization)
		upsample_psi = UpSampling3D(size=(2,2,2))(psi)
		upsample_psi = expend_as(upsample_psi,shape_x[4])
		y = Multiply()([x,upsample_psi])

		result = create_convolution_block(y,shape_x[4],kernel=(1,1,1), activation=None, normalization=normalization)

		return result

	# image input
	inputs = Input(shape=X_shape)

	current_layer = inputs
	level_output_layers = list()
	level_filters = list()
	for level_number in range(depth):
		n_level_filters = min((2**level_number) * n_base_filters,2048)
		level_filters.append(n_level_filters)

		if current_layer is inputs:
			in_conv = create_convolution_block(current_layer, n_level_filters, activation=conv_activation, normalization=normalization)    # in pix2pix-> input layer no instancenorm. this should be turned off for regression? 
		else:
			in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2), activation=conv_activation, normalization=normalization)

		# multipyramid input
		if multi_input and level_number > 1 and level_number < 3:
			pool_size = (2**level_number)
			scaled_input = AveragePooling3D(pool_size=(pool_size,pool_size,pool_size))(inputs)
			conv_scaled_input = create_convolution_block(scaled_input, n_level_filters, activation=conv_activation, normalization=normalization)
			in_conv = concatenate([conv_scaled_input,in_conv], axis=-1)
			in_conv = create_convolution_block(in_conv, n_level_filters, kernel=(1,1,1), activation=conv_activation, normalization=normalization)

		context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

		summation_layer = Add()([in_conv, context_output_layer])
		level_output_layers.append(summation_layer)
		current_layer = summation_layer

	segmentation_layers = list()
	for level_number in range(depth -2, -1, -1):
		up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])

		if use_attention:
			shape = K.int_shape(current_layer)
			g = create_convolution_block(current_layer, shape[4], activation='relu', normalization=normalization)
			attention_layer = create_attention_block(level_output_layers[level_number],g,level_filters[level_number])
			concatenation_layer = concatenate([attention_layer, up_sampling], axis=-1)
		else:
			concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=-1)
		print(concatenation_layer)
		localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
		current_layer = localization_output
		if level_number < n_segmentation_levels:
			segmentation_layers.insert(0, Conv3D(n_labels, (1, 1, 1),kernel_initializer=tf.random_normal_initializer(0., 0.02))(current_layer))

	

	output_layer = None
	for level_number in reversed(range(n_segmentation_levels)):
		segmentation_layer = segmentation_layers[level_number]
		if output_layer is None:
			output_layer = segmentation_layer
		else:
			output_layer = Add()([output_layer, segmentation_layer])
		if level_number > 0:
			output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

	activation_block = Activation(activation_name)(output_layer)

	model = Model(inputs=inputs, outputs=activation_block)

	return model
		
