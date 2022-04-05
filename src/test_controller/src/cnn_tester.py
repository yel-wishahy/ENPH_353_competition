import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2

sess1 = tf.Session()    
graph1 = tf.get_default_graph()
set_session(sess1)

plate_NN = models.load_model("my_model_V3.h5")

def predict_image(image_ar):
	global sess1
	global graph1
	with graph1.as_default():
		set_session(sess1)
		NN_prediction = plate_NN.predict(image_ar)[0]
	return NN_prediction

dummy_image = np.zeros((298,150,3))
dummy_image = cv2.imread("test_image.png")

image_array = np.array([dummy_image])

print(predict_image(image_array))
print(np.argmax(predict_image(image_array)))