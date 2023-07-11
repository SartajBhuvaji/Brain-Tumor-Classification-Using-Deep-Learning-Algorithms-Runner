import os
from flask import Flask, request, send_from_directory, render_template
from werkzeug.utils import secure_filename
import numpy as np
from keras.applications import vgg16
from tensorflow.keras.models import Sequential
from keras.models import Sequential
from keras.layers.core import Dense
import cv2 
from keras.applications import vgg16

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
CATEGORIES = ["glioma_tumor","meningioma_tumor","no_tumor","pituitary_tumor"]
IMG_SIZE = 224
IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)
UPLOAD_FOLDER = 'uploads'

vgg16_model = vgg16.VGG16()
model1 = Sequential()

for layer in vgg16_model.layers[:-3]:
    model1.add(layer) 

for layer in model1.layers:
    layer.trainable = False
    
model1.add(Dense(4,activation = 'softmax'))
model1.load_weights("model/model.h5")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR) 
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def predict(file):
    prediction = model1.predict([prepare(file)])

    if np.argmax(prediction)==0:    
        output="glioma_tumor"
    if np.argmax(prediction)==1:
        output="meningioma_tumor"
    if np.argmax(prediction)==2:
        output="no_tumor"
    if np.argmax(prediction)==3:
        output="pituitary_tumor" 
    return output

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label='', imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)