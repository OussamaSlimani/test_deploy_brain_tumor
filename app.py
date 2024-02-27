app = Flask(__name__)

# Load the trained model only once during the application startup
model_path = "Model.h5"
model = load_model(model_path)

def predict_tumor_class(image_path):
    # Define the classes list
    classes = ['Glioma Tumor', 'Meningioma Tumor', 'Normal', 'Pituitary Tumor']

    # Specify the path to the saved model
    model_path = "Model.h5"

    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)

    # Manually preprocess the image without using TensorFlow's preprocess function
    img_array /= 255.0
    img_array -= 0.5
    img_array *= 2.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.xception.preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(img_array)
