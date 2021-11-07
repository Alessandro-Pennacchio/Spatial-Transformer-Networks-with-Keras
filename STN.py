import numbers

import tensorflow as tf
from stn.transformer import affine_grid_generator, bilinear_sampler
from tensorflow.keras import layers

# Spatial transformer localization-network
def get_localization_network( input_shape ):
	localization = tf.keras.Sequential( [
		layers.Conv2D( 8, kernel_size=7, input_shape=input_shape, activation="relu", kernel_initializer="he_normal" ),
		layers.MaxPool2D( strides=2 ),
		layers.Conv2D( 10, kernel_size=5, activation="relu", kernel_initializer="he_normal" ),
		layers.MaxPool2D( strides=2 ),
	] )
	return localization

# Regressor for the 3 * 2 affine matrix
def get_affine_params():
	fc_loc = tf.keras.Sequential( [
		layers.Flatten(),
		layers.Dense( 32, activation="relu", kernel_initializer="he_normal" ),
		layers.Dense( 3 * 2, kernel_initializer="zeros", bias_initializer=tf.keras.initializers.Constant( [ 1, 0, 0, 0, 1, 0 ] ) )
	] )
	
	return fc_loc

class STN( layers.Layer ):
	"""
	Spatial transformer network forward layer
	"""
	
	def __init__( self, scale=(1, 1), trainable=True, name=None, dynamic=False, **kwargs ):
		super().__init__( trainable, name, dynamic, **kwargs )
		
		if isinstance( scale, numbers.Number ):
			self.scale = (scale, scale)
		elif len( scale ) == 2:
			self.scale = scale
		else:
			raise Exception
		
		self.fc_loc = get_affine_params()
	
	def build( self, input_shape ):
		self.affine_grid_shape = (int( input_shape[ 1 ] * self.scale[ 0 ] ), int( input_shape[ 2 ] * self.scale[ 1 ] ))
		self.localization = get_localization_network( input_shape[ 1: ] )
	
	def call( self, inputs, *args, **kwargs ):  # Defines the computation from inputs to outputs
		
		xs = self.localization( inputs )
		
		theta = self.fc_loc( xs )
		theta = tf.reshape( theta, (-1, 2, 3) )
		
		grid = affine_grid_generator( self.affine_grid_shape[ 0 ], self.affine_grid_shape[ 1 ], theta )
		x_s = grid[ :, 0, :, : ]
		y_s = grid[ :, 1, :, : ]
		x = bilinear_sampler( inputs, x_s, y_s )
		
		return x
	
	def get_config( self ):
		return { "localization": self.localization,
		         "fc_loc": self.fc_loc,
		         "affine_grid_shape": self.affine_grid_shape,
		         }
