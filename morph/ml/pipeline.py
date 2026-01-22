import torch
import torchvision.transforms as transforms

from .models.vae_mnist import MNISTVAE
from .preprocess.digit_single import SingleDigitPreprocessor
from .preprocess.digit_multi import MultiDigitPreprocessor
from .interpolate.linear import linear_interpolate


# Select device (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MorphPipeline:
    """
    Core pipeline for digit morphing.
    Supports:
    - single digit morphing
    - two-digit morphing (pairwise)
    """

    def __init__(self, task="digit"):
        """
        task:
        - "digit"        -> single digit morphing
        - "digit_multi"  -> two digit morphing
        """

        self.task = task

        # Load trained MNIST VAE (shared by both modes)
        self.model = MNISTVAE(latent_dim=10).to(DEVICE)
        self.model.load_state_dict(
            torch.load("mnist_vae.pth", map_location=DEVICE)
        )
        self.model.eval()

        # Image → Tensor
        self.transform = transforms.ToTensor()

        # Choose preprocessor based on task
        if task == "digit":
            self.preprocessor = SingleDigitPreprocessor()

        elif task == "digit_multi":
            self.preprocessor = MultiDigitPreprocessor()

        else:
            raise ValueError("Unsupported task type")

    # --------------------------------------------------
    # SINGLE DIGIT MORPHING
    # --------------------------------------------------
    def run(self, image_a_path, image_b_path, steps):
        """
        Morphs one digit into another
        """

        # Preprocess images
        img_a = self.preprocessor.process(image_a_path)
        img_b = self.preprocessor.process(image_b_path)

        img_a = self.transform(img_a).unsqueeze(0).to(DEVICE)
        img_b = self.transform(img_b).unsqueeze(0).to(DEVICE)

        # Encode to latent space
        with torch.no_grad():
            z1 = self.model.encode_latent(img_a)
            z2 = self.model.encode_latent(img_b)

        # Interpolate in latent space
        latent_vectors = linear_interpolate(z1, z2, steps)

        # Decode interpolated vectors
        frames = []
        with torch.no_grad():
            for z in latent_vectors:
                recon = self.model.decode_latent(z)
                frames.append(recon)

        return frames

    # --------------------------------------------------
    # MULTI DIGIT MORPHING (2 DIGITS)
    # --------------------------------------------------
    def run_multi_digit(self, image_a_path, image_b_path, steps):
        """
        Morphs two digits pairwise.
        Example: 23 → 56
        """

        # Split images into individual digits
        digits_a = self.preprocessor.process(image_a_path)
        digits_b = self.preprocessor.process(image_b_path)

        if len(digits_a) != len(digits_b):
            raise ValueError("Digit count mismatch between images")

        all_frames = []

        # Process each digit independently
        for idx in range(len(digits_a)):
            img_a = self.transform(digits_a[idx]).unsqueeze(0).to(DEVICE)
            img_b = self.transform(digits_b[idx]).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                z1 = self.model.encode_latent(img_a)
                z2 = self.model.encode_latent(img_b)

            latent_vectors = linear_interpolate(z1, z2, steps)

            digit_frames = []
            with torch.no_grad():
                for z in latent_vectors:
                    recon = self.model.decode_latent(z)
                    digit_frames.append(recon)

            # Store frames for this digit
            all_frames.append(digit_frames)

        return all_frames
