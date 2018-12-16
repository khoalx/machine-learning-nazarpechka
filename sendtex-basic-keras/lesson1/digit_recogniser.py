import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 28
IMG_NAME = "img.jpg"


model = tf.keras.models.load_model('epic_num_reader.model')

img_array = cv2.imread(IMG_NAME ,cv2.IMREAD_GRAYSCALE)
img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(img_array, cmap='gray')
plt.show()
img_array = tf.keras.utils.normalize(img_array, axis=1)
prediction = model.predict(img_array.reshape(1,28,28))
print(np.argmax(prediction))
