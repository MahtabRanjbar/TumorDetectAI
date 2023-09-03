from PIL import Image

from controller import TumorDetectionModel
from view import TumorDetectionView


class App:
    def __init__(self, model_path):
        self.model = TumorDetectionModel(model_path)
        self.view = TumorDetectionView()

    def run(self):
        self.view.show_description()
        uploaded_file = self.view.get_uploaded_file()
        if uploaded_file is not None:
            # Load the image
            image = Image.open(uploaded_file)
            # Make a prediction
            label = self.model.predict(image)
            # Display the image and the prediction
            self.view.show_prediction(image, label)


def main():
    controller = App("./saved_models/best_model.h5")
    controller.run()


if __name__ == "__main__":
    main()
