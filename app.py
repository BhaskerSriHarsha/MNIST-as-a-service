from flask import Flask,render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "/home/bhaskersriharsha/Documents/Projects/Papers/0. Genetic rehearsal/MNIST-as-service"


@app.route('/')
def home():
    return render_template('/index.html')

@app.route('/ML',methods=['POST'])
def ML():
    image = request.files['image']
    image.save(os.path.join(app.config["IMAGE_UPLOADS"],"test.jpg"))
    image = cv2.imread('test.jpg',0)
    # image = image/255
    image = np.reshape(image,(1,28,28))
    loaded_model = tf.keras.models.load_model("ML_Model")
    prediction = np.argmax(loaded_model.predict(image), axis=-1)
    print(prediction)
    # prediction = loaded_model.predict_classes(image)
    return render_template('/index.html', model_prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)
