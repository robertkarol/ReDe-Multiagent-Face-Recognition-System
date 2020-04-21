from Services.DirectoryContentVersioner import DirectoryContentVersioner
from Utils.Singleton import SingletonPerKey
from os import path, mkdir, listdir
from typing import Iterable, Tuple


# TODO: Make this process-safe
class NewIdentitiesManager(SingletonPerKey):

    def __new__(cls, *args, **kwargs):
        data_directory = args[0]
        source_locations = args[1]
        instance = super().__new__(cls, data_directory)
        instance.__source_locations = source_locations
        instance.__data_directory = data_directory
        instance.__data_versioner = DirectoryContentVersioner()
        instance.__face_image_filename = "face{}.jpg"
        return instance

    @classmethod
    def get_manager(cls, data_directory: str, source_locations: list):
        return cls.__new__(cls, data_directory, source_locations)

    def get_newest_identities_dataset_path(self, location: str) -> (int, str):
        location_dir = self.__get_location_directory(location)
        data_dir, version = self.__data_versioner.get_latest(location_dir, location)
        identities_count = len(listdir(path.join(data_dir, 'train')))
        if identities_count > 0:
            new_data_dir = path.join(location_dir, self.__data_versioner.get_versioned_name(location, version + 1))
            mkdir(new_data_dir)
            mkdir(path.join(new_data_dir, 'train'))
            mkdir(path.join(new_data_dir, 'test'))
        return identities_count, path.join(location_dir, data_dir)

    def publish_identity(self, location: str, name: str, train_data: Iterable, test_data: Iterable) -> None:
        location_dir = self.__get_location_directory(location)
        data_dir = path.join(location_dir, self.__data_versioner.get_latest(location_dir, location)[0])
        new_id_train_dir, new_id_test_dir = path.join(data_dir, 'train', name), path.join(data_dir, 'val', name)
        mkdir(new_id_train_dir)
        self.__save_images(train_data, new_id_train_dir)
        mkdir(new_id_test_dir)
        self.__save_images(test_data, new_id_test_dir)

    def __save_images(self, images, directory):
        for i, image in enumerate(images):
            image.save(path.join(directory, self.__face_image_filename.format(i)))

    def __get_location_directory(self, location):
        if location not in self.__source_locations:
            raise ValueError(f"{location} is not a valid location")
        return path.join(self.__data_directory, location)
