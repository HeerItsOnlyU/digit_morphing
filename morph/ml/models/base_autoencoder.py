# This is an abstract base class
# Any model (digit VAE, face VAE) must follow this structure

class BaseAutoEncoder:
    def encode(self, image):
        """
        Takes an image and converts it into latent vector
        """
        raise NotImplementedError("Encode method not implemented")

    def decode(self, latent_vector):
        """
        Takes latent vector and reconstructs image
        """
        raise NotImplementedError("Decode method not implemented")
