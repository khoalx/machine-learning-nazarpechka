import cv2
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

CATEGORIES = ["Dog", "Cat"]
IMG_SIZE = 50
FILE_NAME = "shibainu.jpeg"
def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    plt.imshow(new_array, cmap='gray')
    plt.show()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("64x3-CNN.model")
prediction = model.predict([prepare(FILE_NAME)])
print("Prediction for {}".format(FILE_NAME))
print(CATEGORIES[int(prediction[0][0])])
