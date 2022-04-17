from enum import Enum
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
import os
import string

CHARS = string.ascii_uppercase + string.digits

abs_path = os.path.abspath('/home/yel-wishahy/ENPH_353_competition/src/test_controller')

class PreProcessMode(Enum):
	NONE = 0
	GRAY = 1
	BINARY = 2


# model_name = "my_model_V3.h5"
# INPUT_SIZE = (298, 150,3) #h,w, #clr Channels
# PRE_PROCESS_MODE = PreProcessMode.NONE

# model_name = "my_model_V13.h5"
# INPUT_SIZE = (200,100,1) #h,w,#clr channels
#pre_process_mode = PreProcessMode.BINARY

# model_name = "my_model_V15.h5"
# INPUT_SIZE = (100,100,1) #h,w, #clr Channels
# PRE_PROCESS_MODE = PreProcessMode.BINARY

model_name = "ocr_model_2.h5"
INPUT_SIZE = (100, 100,1) #h,w, #clr Channels
PRE_PROCESS_MODE = PreProcessMode.BINARY

# model_name = "my_model_V25.h5"
# INPUT_SIZE = (100,100,1) #h,w, #clr Channels
# PRE_PROCESS_MODE = PreProcessMode.BINARY


model_path = abs_path + '/models/' + model_name

class CharacterDetector():
	def __init__(self):
		self.tf_session = tf.compat.v1.Session() #tf.Session()
		self.tf_graph = tf.compat.v1.get_default_graph() #tf.get_default_graph() 

		set_session(self.tf_session)
		self.model = models.load_model(model_path)

	def predict_image(self,image_ar):
		"""
		@brief uses cnn ocr model to predict a batch of character images

		@param image_ar : array of character images (opencv mat)

		@return list of best prediction tuples : (character, confidence), in order of input 
		"""
		with self.tf_graph.as_default():
			set_session(self.tf_session)
			NN_prediction = self.model.predict(image_ar)

		def process_predictions(predicitons):
			output = []
			for p in predicitons:
				char = CHARS[np.argmax(p)]
				confidence = p.max()
				output.append((char,confidence))
			return output

		return process_predictions(NN_prediction)
	
	@staticmethod
	def prediction_to_string(predictions):
		output = ""
		for p in predictions:
			output+= str(p)
		return output


	




