import h5py
import numpy as np
from PIL import Image, ImageOps
from multiprocessing import Pool

class ImageProcessor:
    def __init__(self):
        pass

    def rotate(self, image, angle):
        if angle != 0:
            image = image.rotate(angle)
        return image

    def resize(self, image, width, height):
        if width > 0 and height > 0:
            image = image.resize((width, height))
        return image

    def grayscale(self, image):
        image = ImageOps.grayscale(image)
        return image

    def invert(self, image):
        image = ImageOps.invert(image)
        return image

    def add_padding(self, image, padding_size, padding_color="#000000"):
        if padding_size > 0:
            image = ImageOps.expand(image, border=padding_size, fill=padding_color)
        return image

    def solarize(self, image, threshold=128):
        image = ImageOps.solarize(image, threshold=threshold)
        return image

    def fit(self, image, width, height):
        if width > 0 and height > 0:
            image = ImageOps.fit(image, (width, height), method=Image.BICUBIC)
        return image

    def flip(self, image):
        image = ImageOps.flip(image)
        return image

    def mirror(self, image):
        image = ImageOps.mirror(image)
        return image

    def get_processed_image(self,image):
        return image

    @staticmethod
    def process_image(image_path, angle=0, width=200, height=200):
        """
        Process a single image using multiprocessing. \n
        :param image_path: Path to the image \n
        :param angle: Rotation angle (default 0) \n
        :param width: Width for resizing (default 200) \n
        :param height: Height for resizing (default 200) \n
        :return: Processed PIL image
        """
        image = Image.open(image_path)
        processor = ImageProcessor(image)
        processed_image = processor.rotate(angle).resize(width, height).grayscale().get_processed_image()
        return processed_image

    @classmethod
    def process_images_parallel(cls, image_paths, angle=0, width=200, height=200):
        """
        Process multiple images in parallel using multiprocessing. \n
        :param image_paths: List of image paths \n
        :param angle: Rotation angle \n
        :param width: Width for resizing \n
        :param height: Height for resizing \n
        :return: List of processed images \n
        """
        with Pool() as pool:
            processed_images = pool.starmap(cls.process_image, [(path, angle, width, height) for path in image_paths])
        return processed_images

class ImageSaver:
    def __init__(self, image_paths, hdf5_file_path, compression="gzip"):
        """
        이미지 데이터를 HDF5로 저장하는 클래스 \n
        :param image_paths: 이미지 파일 경로 목록 \n
        :param hdf5_file_path: 저장할 HDF5 파일 경로 \n
        :param compression: 데이터 압축 방식 (기본: gzip)
        """
        self.image_paths = image_paths
        self.hdf5_file_path = hdf5_file_path
        self.compression = compression

    def process_image(self, image_path):
        """
        이미지를 읽고 배열로 변환하는 함수 \n
        :param image_path: 이미지 파일 경로 \n
        :return: 이미지 배열
        """
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_array = np.array(img)
        return img_array
    
    def save_images_to_hdf5_parallel(self, dset_name):
        """
        이미지 데이터 병렬처리 및 HDF5 포맷으로 저장
        """

        # with Pool() as pool:
        #     images = pool.map(self.process_image, self.image_paths)
        images = self.process_image(self.image_paths)
        
        with h5py.File(self.hdf5_file_path, 'w') as hf:
            images_dataset = hf.create_dataset(
                dset_name,
                data=images, #np.array(images),
                compression=self.compression, # 압축 방식 설정
                compression_opts=5,  # 압축 레벨 (0-9, 9가 최고 압축)
                dtype='uint8' # 이미지 데이터는 보통 uint8로 저장됨
            )
            print(f"데이터가 {self.hdf5_file_path}에 저장되었습니다.")

class ImageReader:
    def __init__(self, hdf5_file_path):
        """
        hdf5 파일 경로 입력
        """
        self.hdf5_file_path = hdf5_file_path

    def read_img(self, hdf5_path):
        """
        hdf5 파일 읽기
        """
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            if hdf5_path in hf:
                dset = hf[hdf5_path]
                data = dset[:]
                return data
            else:
                raise KeyError(f"Dataset '{hdf5_path}' not found in the file.")
            
    def list_datasets(self):
        """
        hdf5 파일 안에 있는 모든 데이터셋 리스트(List) 출력
        """
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            return list(hf.keys())
        
    def read_all_img(self):
        """
        hdf5 파일 안에 있는 모든 데이터 출력
        """
        all_data = {}
        with h5py.File(self.hdf5_file_path, 'r') as hf:
            for dset_name in hf.keys():
                all_data[dset_name] = hf[dset_name][:]
        return all_data