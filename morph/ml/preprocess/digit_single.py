from .base_preprocess import BasePreprocessor
from PIL import Image

class SingleDigitPreprocessor(BasePreprocessor):
    def process(self, image_path):
        # Load image
        img = Image.open(image_path).convert("L")

        # Resize to MNIST size
        img = img.resize((28, 28))

        return img
