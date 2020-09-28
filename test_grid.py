import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np

print("Import successful")

# Create the model
model = Sequential()
model.add(Flatten())
model.add(Dense(784,activation="relu"))
model.add(Dense(200,activation="relu"))
model.add(Dense(10,activation="softmax"))
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(optimizer="adam", loss=loss_fn)

print("Model creation successful")

# Import the dataset and clean it
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_train.shape)

np.save("sample.npy",x_train[0])

# Train the model
history_model = model.fit(x_train,y_train, epochs=5, batch_size=100, verbose=1)
print(" ")
model.evaluate(x_test,y_test,verbose=1)

# Save the model to the disk
model.save("ML_Model")

print("Model saved")

loaded_model = tf.keras.models.load_model("ML_Model")
print("Model loaded from disk")

loaded_model.evaluate(x_test,y_test,verbose=1)
