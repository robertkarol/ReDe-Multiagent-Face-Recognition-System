from os import listdir, mkdir, path
from os.path import isdir
from PIL import Image
from matplotlib import pyplot
from mtcnn import MTCNN
from numpy import asarray

class DatasetHelpers:
    @staticmethod
    def extract_faces_from_image(path_to_image, required_size = (160, 160), single_face = False):
        image_as_pixels = DatasetHelpers.image_from_path_to_pixels_array(path_to_image)
        face_detector = MTCNN()

        results = face_detector.detect_faces(image_as_pixels)
        no_of_faces_to_extract = 1 if single_face else len(results)
        face_array_list = []
        for i in range(no_of_faces_to_extract):
            x1, y1, width, height = results[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face = image_as_pixels[y1:y2, x1:x2]
            image = Image.fromarray(face)
            face_array = DatasetHelpers.image_to_pixels_array(image, required_size)
            face_array_list.append(face_array)

        return face_array_list


    @staticmethod
    def extract_faces_from_directory(processed_folder, extraction_folder = None, extracted_face_file_prefix ="face",
                                     face_file_extension =".jpg", single_face = False):
        i = 1
        extracted_faces = []

        for filename in listdir(processed_folder):
            path = processed_folder + "/" + filename
            face_array_list = DatasetHelpers.extract_faces_from_image(path, single_face=single_face)
            if extraction_folder:
                for face_array in face_array_list:
                    pyplot.imsave(extraction_folder + "/" + extracted_face_file_prefix + str(i) + face_file_extension, face_array)
                    i += 1
            else:
                extracted_faces.extend(face_array_list)

        if not extraction_folder: return extracted_faces


    @staticmethod
    def extract_faces_from_dataset(dataset_folder, extraction_folder, create_extraction_folder = True, single_face = False):
        extraction_folder = dataset_folder + "-extracted" if not extraction_folder else extraction_folder
        extracted_faces = []

        if create_extraction_folder: mkdir(extraction_folder)
        for phase in listdir(dataset_folder):
            if create_extraction_folder: mkdir(extraction_folder + "/" + phase)
            for subject in listdir(dataset_folder + "/" + phase):
                if create_extraction_folder: mkdir(extraction_folder + "/" + phase + "/" + subject)
                DatasetHelpers.extract_faces_from_directory(dataset_folder + "/" + phase + "/" + subject,
                                                            extraction_folder + "/" + phase + "/" + subject,
                                                            single_face=single_face)


    @staticmethod
    def image_from_path_to_pixels_array(path_to_image, required_size = None):
        image = Image.open(path_to_image)

        return DatasetHelpers.image_to_pixels_array(image, required_size)


    @staticmethod
    def image_to_pixels_array(image, required_size = None):
        image = image.convert('RGB')
        if required_size: image = image.resize(required_size)
        return asarray(image)


    @staticmethod
    def load_images(directory):
        images = []
        for filename in listdir(directory):
            path_to_file = directory + filename
            if not path.isfile(path_to_file): continue
            image = Image.open(path_to_file)
            images.append(asarray(image))
        return images


    @staticmethod
    def load_single_dataset(directory):
        input_data, output_data = [], []
        i = 1
        for subdir in sorted(listdir(directory)):
            path = directory + subdir + '/'
            if not isdir(path): continue
            faces = DatasetHelpers.load_images(path)
            labels = [i] * len(faces)
            input_data.extend(faces)
            output_data.extend(labels)
            i += 1
        return (asarray(input_data), asarray(output_data))


    @staticmethod
    def load_datasets(datasets_directory):
        datasets = []
        for directory in listdir(datasets_directory):
            datasets.append(DatasetHelpers.load_single_dataset(directory))
        return datasets