
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
#from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np
from keras.applications import vgg16
from tensorflow.keras.models import Sequential

from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
import keras.optimizers

from keras.applications import vgg16

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (224, 224)#150,150 old
UPLOAD_FOLDER = 'uploads'

vgg16_model = vgg16.VGG16()
model1 = Sequential()
#for layer in vgg16_model.layers:
for layer in vgg16_model.layers[:-1]: #model1.layers.pop() not working, so -1
    model1.add(layer) # convert to sequantial model

for layer in model1.layers:
    layer.trainable = False
    
model1.add(Dense(4,activation = 'softmax'))
model1.load_weights("model/weights.h5")

IMG_SIZE = 224

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


import cv2 # image size
CATEGORIES = ["glioma","meningioma","no_tumor","pituitary"]

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

def predict(file):
   # img  = load_img(file, target_size=IMAGE_SIZE)
    #img = img_to_array(img)/255.0
    #img = np.expand_dims(img, axis=0)
    #probs = vgg16.predict(img)[0]
    prediction = model1.predict([prepare(file)])
    #output = CATEGORIES[int(prediction[0][0])]
    if np.argmax(prediction)==0:    
        output="glioma"
    if np.argmax(prediction)==1:
        output="meningioma"
    if np.argmax(prediction)==2:
        output="no_tumor"
    if np.argmax(prediction)==3:
        output="pituitary" 
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
            output = predict(file_path) #call predict fn
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
