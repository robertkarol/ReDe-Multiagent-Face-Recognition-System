from Services.DirectoryContentVersioner import DirectoryContentVersioner
from Services.RecognitionLocationsManager import RecognitionLocationsManager
from Utils.Singleton import SingletonPerKey
from multiprocessing import Lock
from os import path, mkdir, listdir
from typing import Iterable


class NewIdentitiesManager(SingletonPerKey):

    def __new__(cls, *args, **kwargs):
        data_directory = args[0]
        instance = super().__new__(cls, data_directory)
        instance.__source_locations_manager = RecognitionLocationsManager()
        instance.__data_directory = data_directory
        instance.__data_versioner = DirectoryContentVersioner()
        instance.__face_image_filename = "face{}.jpg"
        instance.__lock = Lock()
        return instance

    @classmethod
    def get_manager(cls, data_directory: str):
        return cls.__new__(cls, data_directory)

    def get_newest_identities_dataset_path(self, location: str) -> (int, str):
        location_dir = self.__get_location_directory(location)
        if not path.isdir(location_dir):
            raise LookupError(f"No such {location_dir} identities source being versioned")

        data_dir, version = self.__data_versioner.get_latest(location_dir, location)
        if not data_dir:
            return 0, None
        try:
            identities_count = len(listdir(path.join(location_dir, data_dir, 'train')))
        except FileNotFoundError as e:
            print(e)
            return 0, None

        if identities_count > 0:
            self.__lock.acquire()
            try:
                new_data_dir = path.join(location_dir, self.__data_versioner.get_versioned_name(location, version + 1))
                mkdir(new_data_dir)
                mkdir(path.join(new_data_dir, 'train'))
                mkdir(path.join(new_data_dir, 'val'))
            finally:
                self.__lock.release()

        return identities_count, path.join(location_dir, data_dir)

    def publish_identity(self, location: str, name: str, train_data: Iterable, test_data: Iterable) -> None:
        location_dir = self.__get_location_directory(location)
        self.__lock.acquire()
        try:
            data_dir = path.join(location_dir, self.__data_versioner.get_latest(location_dir, location)[0])
            new_id_train_dir, new_id_test_dir = path.join(data_dir, 'train', name), path.join(data_dir, 'val', name)
            mkdir(new_id_train_dir)
            self.__save_images(train_data, new_id_train_dir)
            mkdir(new_id_test_dir)
            self.__save_images(test_data, new_id_test_dir)
        finally:
            self.__lock.release()

    def get_recognition_locations(self):
        return self.__source_locations_manager.get_recognition_locations()

    def __save_images(self, images, directory):
        for i, image in enumerate(images):
            image.save(path.join(directory, self.__face_image_filename.format(i)), 'JPEG')

    def __get_location_directory(self, location):
        if location not in self.__source_locations_manager.get_recognition_locations():
            raise ValueError(f"{location} is not a valid location")
        return path.join(self.__data_directory, location)
