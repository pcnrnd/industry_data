from PIL import Image, ImageOps
from multiprocessing import Pool

class ImageProcessor:
    def __init__(self, image):
        """
        Initialize the ImageProcessor with an image.
        :param image: PIL Image object to process
        """
        self.image = image

    def rotate(self, angle):
        if angle != 0:
            self.image = self.image.rotate(angle)
        return self

    def resize(self, width, height):
        if width > 0 and height > 0:
            self.image = self.image.resize((width, height))
        return self

    def grayscale(self):
        self.image = ImageOps.grayscale(self.image)
        return self

    def invert(self):
        self.image = ImageOps.invert(self.image)
        return self

    def add_padding(self, padding_size, padding_color="#000000"):
        if padding_size > 0:
            self.image = ImageOps.expand(self.image, border=padding_size, fill=padding_color)
        return self

    def solarize(self, threshold=128):
        self.image = ImageOps.solarize(self.image, threshold=threshold)
        return self

    def fit(self, width, height):
        if width > 0 and height > 0:
            self.image = ImageOps.fit(self.image, (width, height), method=Image.BICUBIC)
        return self

    def flip(self):
        self.image = ImageOps.flip(self.image)
        return self

    def mirror(self):
        self.image = ImageOps.mirror(self.image)
        return self

    def get_processed_image(self):
        return self.image

    @staticmethod
    def process_image(image_path, angle=0, width=200, height=200):
        """
        Process a single image using multiprocessing.
        :param image_path: Path to the image
        :param angle: Rotation angle (default 0)
        :param width: Width for resizing (default 200)
        :param height: Height for resizing (default 200)
        :return: Processed PIL image
        """
        image = Image.open(image_path)
        processor = ImageProcessor(image)
        processed_image = processor.rotate(angle).resize(width, height).grayscale().get_processed_image()
        return processed_image

    @classmethod
    def process_images_parallel(cls, image_paths, angle=0, width=200, height=200):
        """
        Process multiple images in parallel using multiprocessing.
        :param image_paths: List of image paths
        :param angle: Rotation angle
        :param width: Width for resizing
        :param height: Height for resizing
        :return: List of processed images
        """
        with Pool() as pool:
            processed_images = pool.starmap(cls.process_image, [(path, angle, width, height) for path in image_paths])
        return processed_images

