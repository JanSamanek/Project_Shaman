import tensorflow.keras.backend as K

def euclidean_distance(vectors):
	(features_1, features_2) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(features_1 - features_2), axis=1, keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))
