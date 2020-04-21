from Utils.Singleton import SingletonMeta


class NewIdentitiesManager(metaclass=SingletonMeta):
    def __init__(self, source_locations: list, data_directory):
        self.__source_locations = source_locations
        self.__data_directory = data_directory

    def get_new_identities_dataset_path(self, location: str) -> (int, str):
        pass

    def publish_identity(self, location: str, name: str, train_data, test_data) -> None:
        pass
