# coding=utf-8
import os
import glob
import numpy as np

# Keras
import tensorflow as tf
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template

# Define a flask app
app = Flask(__name__)

# Use Keras pre-trained mobilenet to classify images
from keras.applications.mobilenet import MobileNet, preprocess_input
model = MobileNet(weights='imagenet')
print('Model loaded'')

# need to get tensorflow on the same thread as flask
global graph
graph = tf.get_default_graph()


def model_predict(img, model):
    img = image.load_img(img, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    with graph.as_default():
        preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Make prediction
        preds = model_predict(f, model)

        # Process your result for human
        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string
    return render_template('predict.html', result=result)


if __name__ == '__main__':
    app.run(port=5002, debug=True)
