from keras.models import load_model
import numpy as np
from numpy import expand_dims, load, asarray
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from DatasetHelpers import DatasetHelpers
from ResourceLocalizer import ResourceLocalizer
import pickle

class RecognitionModel:

    def __init__(self, normalize, encode_labels, model_type='knn'):
        self.__resource_localizer = ResourceLocalizer()
        self.__embeddings_model = load_model(self.__resource_localizer.FaceNetModel)
        self.__model_type = model_type
        if model_type == 'knn':
            self.__classification_model = KNeighborsClassifier(n_jobs=-1)
        elif model_type == 'svm':
            self.__classification_model = SVC(kernel='linear', probability=True)
        else:
            raise ValueError("Invalid model type: " + model_type)
        self.__input_normalizer = Normalizer(norm='l2') if normalize else None
        self.__output_encoder = LabelEncoder() if encode_labels else None


    def load_data_from_compressed(self, path_to_file, is_embeddings_file):
        data = load(path_to_file)
        self.__train_input, self.__train_output, self.__test_input, self.__test_output = data['arr_0'], data['arr_1'], \
                                                                                         data['arr_2'], data['arr_3']
        if not is_embeddings_file:
            self.__train_input = self.__get_embedded_dataset(self.__train_input)
            self.__test_input = self.__get_embedded_dataset(self.__test_input)
        print(self.__train_input)


    def load_data_from_directory(self, dataset_path, append=False):
        data = DatasetHelpers.load_datasets(dataset_path)
        if append:
            label_offset = self.__train_output[-1]
            self.__train_input = np.concatenate((self.__train_input, self.__get_embedded_dataset(data[0][0])))
            self.__test_input = np.concatenate((self.__test_input, self.__get_embedded_dataset(data[1][0])))
            self.__train_output = np.concatenate((self.__train_output, data[0][1] + label_offset))
            self.__test_output = np.concatenate((self.__test_output, data[1][1] + label_offset))
        else:
            self.__train_input, self.__train_output, self.__test_input, self.__test_output = data[0][0], data[0][1], \
            self.__train_input = self.__get_embedded_dataset(self.__train_input)
            self.__test_input = self.__get_embedded_dataset(self.__test_input)


    def train(self, neighbors=1):
        self.__neighbors = neighbors
        self.__train(neighbors)


    def __train(self, neighbors=1, transform_data=True):
        train_input, train_output = self.__train_input, self.__train_output
        if transform_data: train_input, train_output = self.__transform_data(train_input, train_output)
        if self.model_type == 'knn':
            self.__classification_model.set_params(n_neighbors=neighbors)
        self.__classification_model.fit(train_input, train_output)


    def retrain_from_dataset(self, additional_dataset_path):
        self.load_data_from_directory(additional_dataset_path, append=True)
        self.train(neighbors=self.__neighbors)


    def test(self):
        test_input, test_output = self.__transform_data(self.__test_input, self.__test_output)
        prediction = self.__classification_model.predict(test_input)
        score_test = accuracy_score(test_output, prediction)
        return score_test


    def test_threshold(self, threshold):
        test_input, test_output = self.__transform_data(self.__test_input, self.__test_output)
        prediction_probabilities = self.__predict_from_faces_embeddings(test_input, is_array=True, transform_data=False)
        prediction = []
        classified = classified_correctly = 0
        total_predictions = len(prediction_probabilities)
        for i in range(total_predictions):
            if prediction_probabilities[i][1] > threshold:
                cls = prediction_probabilities[i][0]
                classified += 1
                classified_correctly += 1 if cls == test_output[i] else 0
            else:
                cls = -1
            prediction.append(cls)

        return classified_correctly / classified, classified / total_predictions


    def predict_from_faces_embeddings(self, faces_embeddings_list, is_array=False):
        return self.__predict_from_faces_embeddings(faces_embeddings_list, is_array)


    def __predict_from_faces_embeddings(self, faces_embeddings_list, is_array=False, transform_data=True):
        if not is_array: faces_embeddings_list = asarray(faces_embeddings_list)
        if transform_data: faces_embeddings_list, _ = self.__transform_data(faces_embeddings_list, asarray([]))
        prediction_probabilities = self.__classification_model.predict_proba(faces_embeddings_list)
        predictions = []
        for instance in prediction_probabilities:
            predicted_class = np.argmax(instance) + 1
            predictions.append((predicted_class, instance[predicted_class - 1]))
        return predictions


    def predict_from_faces_pixels(self, faces_pixels_list):
        faces_embeddings_list = [self.__get_embedding(face_pixels) for face_pixels in faces_pixels_list]
        return self.predict_from_faces_embeddings(faces_embeddings_list)


    def predict_from_faces_images(self, face_images_list):
        faces_pixels_list = [DatasetHelpers.image_to_pixels_array(face_image, (160, 160)) for face_image in
                             face_images_list]
        return self.predict_from_faces_pixels(faces_pixels_list)


    @property
    def model_type(self):
        return self.__model_type


    @staticmethod
    def save_model_as_binary(recognition_model, filename):
        with open(filename + ".pkl", "wb") as output:
            pickle.dump(recognition_model, output, pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load_model_from_binary(path_to_binary):
        with open(path_to_binary + ".pkl", "rb") as input:
            return pickle.load(input)


    def get_embedding(self, face_image):
        return self.__get_embedding(DatasetHelpers.image_to_pixels_array(face_image, (160, 160)))


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


    def __get_embedded_dataset(self, input):
        embedded_input = []

        for face_pixels in input:
            embedding = self.__get_embedding(face_pixels)
            embedded_input.append(embedding)
        embedded_input = asarray(embedded_input)

        return embedded_input