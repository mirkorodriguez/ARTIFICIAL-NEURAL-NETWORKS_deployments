#Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
#Import Keras
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
#Import python files
from inceptionV3_loader import cargarModeloInceptionV3
import numpy as np

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

#Define a route
@app.route('/')
def default():
    return 'TensorFlow Serving ... Go to /<model>/predict'

@app.route('/inception/predict/',methods=['POST'])
def predict():

    model_name = "inception"

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
            print("filename:",filename)

            img = image.img_to_array(image.load_img(filename, target_size=(224, 224)))
            img = img.astype('float32')

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

# Run de application
app.run(host='0.0.0.0',port=5000)
