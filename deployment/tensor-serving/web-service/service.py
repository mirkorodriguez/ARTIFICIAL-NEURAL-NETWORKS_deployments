#Reference: https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
#Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
#Import Keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
#Import python files
import numpy as np

import requests
import json
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '../../../samples/images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Initialize the application service
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funciones
def port(x):
    return {
        'inception': '8501',
        'vgg': '8502',
        'resnet': '8503',
        'mobilenet': '8504',
    }[x]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(model_name):
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading image
            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)

            img = image.img_to_array(image.load_img(filename, target_size=(224, 224)))

            if (model_name == 'inception' or model_name == 'mobilenet'):
                img = img / 255.

            img = img.astype('float16')

            payload = {"instances": [{'input_image': img.tolist()}]}

            # URI
            uri = ''.join(['http://localhost:',port(model_name),'/v1/models/',model_name,':predict'])
            print("URI:",uri)

            # Request al modelo desplegado en TensorFlow Serving
            r = requests.post(uri, json=payload)
            pred = json.loads(r.content.decode('utf-8'))

            # Decodificando decoder util
            predictions = decode_predictions(np.array(pred['predictions']),top=1)
            print("Predictions:\n",predictions)
            label = predictions[0][0][1]
            score = predictions[0][0][2]

            #Results as Json
            data["predictions"] = []
            r = {"label": label, "score": float(score)}
            data["predictions"].append(r)

            #Success
            data["success"] = True

            return jsonify(data)


#Define a route
@app.route('/')
def default():
    return 'TensorFlow Serving ... Go to /<model>/predict'

# VGG
@app.route('/vgg/predict/',methods=['POST'])
def vgg():
    model_name = "vgg"
    return (predict(model_name))

# Inception V3
@app.route('/inception/predict/',methods=['POST'])
def inception():
    model_name = "inception"
    return (predict(model_name))

# ResNet
@app.route('/resnet/predict/',methods=['POST'])
def resnet():
    model_name = "resnet"
    return (predict(model_name))

# MobileNet
@app.route('/mobilenet/predict/',methods=['POST'])
def mobilenet():
    model_name = "mobilenet"
    return (predict(model_name))

# Run de application
app.run(host='0.0.0.0',port=5000)
