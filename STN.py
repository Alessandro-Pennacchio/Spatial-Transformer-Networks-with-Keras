import numbers

import tensorflow as tf
from stn.transformer import affine_grid_generator, bilinear_sampler
from tensorflow.keras import layers, Model

def get_localization_network( input_shape ):
	localization = tf.keras.Sequential( [
		
		# Spatial transformer localization-network
		layers.Conv2D( 8, kernel_size=7, input_shape=input_shape, activation="relu", kernel_initializer="he_normal" ),
		layers.MaxPool2D( strides=2 ),
		layers.Conv2D( 10, kernel_size=5, activation="relu", kernel_initializer="he_normal" ),
		layers.MaxPool2D( strides=2 ),
		
		# Regressor for the 3 * 2 affine matrix
		layers.Flatten(),
		layers.Dense( 32, activation="relu", kernel_initializer="he_normal" ),
		layers.Dense( 3 * 2, kernel_initializer="zeros", bias_initializer=tf.keras.initializers.Constant( [ 1, 0, 0, 0, 1, 0 ] ) )
	] )
	return localization

class STN( layers.Layer ):
	"""
	Spatial transformer network forward layer
	"""
	
	def __init__( self, scale=(1, 1), localizationNetwork=None, **kwargs ):
		
		self.localizationNetwork_set = False
		
		if localizationNetwork is not None:
			assert isinstance( localizationNetwork, (layers.Layer, Model) )
			assert tf.reduce_prod( localizationNetwork.output_shape[ 1: ] ) == 6
			self.localizationNetwork = localizationNetwork
			self.localizationNetwork_set = True
		
		super().__init__( **kwargs )
		
		if isinstance( scale, numbers.Number ):
			self.scale = (scale, scale)
		elif len( scale ) == 2:
			self.scale = scale
		else:
			raise Exception
	
	def build( self, input_shape ):
		self.affine_grid_shape = (int( input_shape[ 1 ] * self.scale[ 0 ] ), int( input_shape[ 2 ] * self.scale[ 1 ] ))
		
		if not self.localizationNetwork_set:
			self.localizationNetwork = get_localization_network( input_shape[ 1: ] )
			self.localizationNetwork_set = True
		# else: # todo: causes trouble during load but it is "advisable" for customs localizationNetwork
		# 	assert input_shape == self.localizationNetwork.input_shape
	
	def call( self, inputs, *args, **kwargs ):  # Defines the computation from inputs to outputs
		
		theta = self.localizationNetwork( inputs )
		
		theta = tf.reshape( theta, (-1, 2, 3) )
		
		grid = affine_grid_generator( self.affine_grid_shape[ 0 ], self.affine_grid_shape[ 1 ], theta )
		x_s = grid[ :, 0, :, : ]
		y_s = grid[ :, 1, :, : ]
		x = bilinear_sampler( inputs, x_s, y_s )
		
		return x
	
	def get_config( self ):
		config = super().get_config()
		config.update( { "scale": self.scale,
		                 "localizationNetwork": self.localizationNetwork,
		                 # "affine_grid_shape": self.affine_grid_shape, # not needed
		                 } )
		return config
