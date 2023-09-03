import numpy as np
from tensorflow.keras.models import load_model


class TumorDetectionModel:
    def __init__(self, model_path):
        # Load the model
        self.model = load_model(model_path)

    def predict(self, image):
        # Preprocess the image
        image = self.preprocess_image(image)
        # Make a prediction
        prediction = self.model.predict(image)
        # Get the prediction label
        label = self.get_prediction_label(prediction)
        return label

    def preprocess_image(self, image):
        # Resize the image
        image = image.resize((224, 224))
        # Convert the image to an array
        img_array = np.array(image)
        # Normalize the pixel values
        # img_array = img_array / 255.0
        # Expand the dimensions of the array
        img_array = np.expand_dims(img_array, axis=0)
        # Return the preprocessed image
        return img_array

    def get_prediction_label(self, prediction):
        print(prediction)
        pred = np.argmax(prediction, axis=1)
        print(pred)
        print(prediction)
        if pred == 0:
            return "No Tumor"
        else:
            return "Tumor"
