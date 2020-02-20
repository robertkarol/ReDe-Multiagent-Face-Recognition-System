from keras.models import load_model
from numpy import expand_dims, load, asarray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from DatasetHelpers import DatasetHelpers
from ResourceLocalizer import ResourceLocalizer
import pickle

class RecognitionModel:

    def __init__(self, normalize, encode_labels):
        self.__resource_localizer = ResourceLocalizer()
        self.__embeddings_model = load_model(self.__resource_localizer.FaceNetModel)
        self.__classification_model = SVC(kernel='linear', probability=True)
        self.__input_normalizer = Normalizer(norm='l2') if normalize else None
        self.__output_encoder = LabelEncoder() if encode_labels else None


    def load_data_from_compressed(self, path_to_file, is_embeddings_file):
        data = load(path_to_file)
        self.__train_input, self.__train_output, self.__test_input, self.__test_output = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
        if not is_embeddings_file:
            embedded_train_input = []
            for face_pixels in self.__train_input:
                embedding = self.__get_embedding(face_pixels)
                embedded_train_input.append(embedding)
            embedded_train_input = asarray(embedded_train_input)

            embedded_test_input = []
            for face_pixels in self.__test_input:
                embedding = self.__get_embedding(face_pixels)
                embedded_test_input.append(embedding)
            embedded_test_input = asarray(embedded_test_input)
            self.__train_input, self.__train_output = embedded_train_input, embedded_test_input


    def load_data_from_directory(self, dataset_path):
        data = DatasetHelpers.load_datasets(dataset_path)
        self.__train_input, self.__train_output, self.__test_input, self.__test_output = data[0][0], data[0][1], data[1][0], data[1][1]


    def train(self):
        train_input, train_output = self.__transform_data(self.__train_input, self.__train_output)
        self.__classification_model.fit(train_input, train_output)


    def test(self):
        test_input, test_output = self.__transform_data(self.__test_input, self.__test_output)
        prediction = self.__classification_model.predict(test_input)
        score_test = accuracy_score(test_output, prediction)
        return score_test


    def predict_from_face_arrays(self, faces_as_array_list):
        return self.__classification_model.predict(asarray(faces_as_array_list))


    def predict_from_face_images(self, face_images_list):
        faces_as_array_list = [DatasetHelpers.image_to_pixels_array(face_image, (160, 160)) for face_image in face_images_list]
        return self.predict_from_face_arrays(faces_as_array_list)


    @staticmethod
    def save_model_as_binary(recognition_model, filename):
        with open(filename + ".pkl", "wb") as output:
            pickle.dump(recognition_model, output, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_model_from_binary(path_to_binary):
        with open(path_to_binary + ".pkl", "rb") as input:
            return pickle.load(input)


    def __transform_data(self, input, output):
        input = self.__input_normalizer.transform(input) if self.__input_normalizer else input
        if self.__output_encoder:
            self.__output_encoder.fit(output)
            output = self.__output_encoder.transform(output)
        return input, output


    def __get_embedding(self, face_pixels):
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        embedding = self.__embeddings_model.predict(samples)
        return embedding[0]