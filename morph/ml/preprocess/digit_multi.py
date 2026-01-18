import cv2
from PIL import Image
import numpy as np

class MultiDigitPreprocessor:
    def process(self, image_path):
        """
        Splits a 2-digit image into two single-digit images
        Assumption: digits are side-by-side
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        h, w = img.shape
        mid = w // 2

        digit1 = img[:, :mid]
        digit2 = img[:, mid:]

        digit1 = cv2.resize(digit1, (28, 28))
        digit2 = cv2.resize(digit2, (28, 28))

        digit1 = Image.fromarray(digit1)
        digit2 = Image.fromarray(digit2)

        return [digit1, digit2]
