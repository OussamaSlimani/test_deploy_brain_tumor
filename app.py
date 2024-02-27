from flask import Flask, render_template, request
from flask_caching import Cache
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

def load_model_if_necessary():
    # Load the model if it's not already in the cache
    if not cache.get('model'):
        # Define the classes list
        classes = ['Glioma Tumor', 'Meningioma Tumor', 'Normal', 'Pituitary Tumor']

        # Specify the path to the saved model
        model_path = "Model.h5"

        # Load the trained model
        model = load_model(model_path)
        
        # Cache the model for subsequent requests
        cache.set('model', model, timeout=None)
        cache.set('classes', classes, timeout=None)

@cache.memoize(timeout=60)  # Cache the prediction result for 60 seconds
def predict_tumor_class(image_path):
    # Load the model if not already in the cache
    load_model_if_necessary()

    # Retrieve the model from the cache
    model = cache.get('model')
    classes = cache.get('classes')

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)

    # Decode predictions
    class_index = np.argmax(predictions)
    class_label = classes[class_index]

    return class_label

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded file
        imagefile = request.files['imagefile']

        # Ensure the "images" folder exists
        os.makedirs("images", exist_ok=True)

        image_path = "./images/" + imagefile.filename
        imagefile.save(image_path)

        predicted_class = predict_tumor_class(image_path)

        return render_template('index.html', prediction=predicted_class)

    return render_template("index.html")

if __name__ == '__main__':
    app.run(port=3000, debug=True)
