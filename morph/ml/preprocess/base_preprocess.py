class BasePreprocessor:
    def process(self, image_path):
        """
        Loads and preprocesses image
        """
        raise NotImplementedError("Process method not implemented")
