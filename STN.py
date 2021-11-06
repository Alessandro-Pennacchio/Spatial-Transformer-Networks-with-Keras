import tensorflow as tf
import tensorflow.python.keras.engine.base_layer
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

output_bias = tf.keras.initializers.Constant( [ 1, 0, 0, 0, 1, 0 ] )

# Regressor for the 3 * 2 affine matrix
def get_affine_params():
	fc_loc = tf.keras.Sequential( [
		layers.Dense( 32, activation="relu", kernel_initializer="he_normal" ),
		layers.Dense( 3 * 2, kernel_initializer="zeros", bias_initializer=output_bias )
	] )
	
	return fc_loc

class STN( tensorflow.python.keras.engine.base_layer.Layer ):
	"""
	Spatial transformer network forward layer
	"""
	
	def __init__( self, scale, trainable=True, name="Spatial_Transformer_Network", dtype=None, dynamic=False, **kwargs ):
		super().__init__( trainable, name, dtype, dynamic, **kwargs )
		self.scale = scale
		self.fc_loc = get_affine_params()
	
	def build( self, input_shape ):
		self.localization = get_localization_network( input_shape[ 1: ] )
	
	def call( self, inputs, *args, **kwargs ):  # Defines the computation from inputs to outputs
		
		xs = self.localization( inputs )
		xs = tf.reshape( xs, (-1, tf.math.reduce_prod( xs.shape[ 1: ] )) )
		
		theta = self.fc_loc( xs )
		theta = tf.reshape( theta, (-1, 2, 3) )
		
		grid = affine_grid_generator( int( inputs.shape[ 1 ] * self.scale ), int( inputs.shape[ 2 ] * self.scale ), theta )
		x_s = grid[ :, 0, :, : ]
		y_s = grid[ :, 1, :, : ]
		x = bilinear_sampler( inputs, x_s, y_s )
		
		return x
	
	def get_config( self ):
		return { "localization": self.localization,
		         "fc_loc": self.fc_loc,
		         }
