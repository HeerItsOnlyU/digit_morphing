import torch
from PIL import Image
import torchvision.transforms as transforms

from .models.vae_mnist import MNISTVAE
from .preprocess.digit_single import SingleDigitPreprocessor
from .interpolate.linear import linear_interpolate
from .preprocess.digit_multi import MultiDigitPreprocessor



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MorphPipeline:
    def __init__(self, task="digit"):
        self.task = task

        if task == "digit":
            self.model = MNISTVAE(latent_dim=10).to(DEVICE)
            self.model.load_state_dict(torch.load("mnist_vae.pth", map_location=DEVICE))
            self.model.eval()

            self.preprocessor = SingleDigitPreprocessor()
            self.transform = transforms.ToTensor()
        else:
            raise ValueError("Only digit task supported for now")

    def run(self, image_a_path, image_b_path, steps):
        # ---------- Preprocess ----------
        img_a = self.preprocessor.process(image_a_path)
        img_b = self.preprocessor.process(image_b_path)

        img_a = self.transform(img_a).unsqueeze(0).to(DEVICE)
        img_b = self.transform(img_b).unsqueeze(0).to(DEVICE)

        # ---------- Encode ----------
        with torch.no_grad():
            z1 = self.model.encode_latent(img_a)
            z2 = self.model.encode_latent(img_b)

        # ---------- Interpolate ----------
        latent_vectors = linear_interpolate(z1, z2, steps)

        # ---------- Decode ----------
        frames = []
        with torch.no_grad():
            for z in latent_vectors:
                recon = self.model.decode_latent(z)
                frames.append(recon)

        return frames
