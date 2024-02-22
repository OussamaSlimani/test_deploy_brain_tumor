from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='converted_model_vgg.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def names(number):
    if number == 0:
        return 'Glioma Tumor'
    elif number == 1:
        return 'Meningioma Tumor'
    elif number == 2:
        return 'No Tumor'
    elif number == 3:
        return 'Pituitary Tumor'
    else:
        return 'unknown'

@app.route('/', methods=['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded file
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Load and resize the image
    img = image.load_img(image_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Preprocess the image and make predictions using TensorFlow Lite Interpreter
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    res = interpreter.get_tensor(output_details[0]['index'])

    # Get the classification with the highest probability
    classification = np.argmax(res)

    return render_template('index.html', prediction=names(classification))

if __name__ == '__main__':
    app.run(port=3000, debug=True)
