from Utils.Singleton import SingletonMeta
import configparser


class ResourceLocalizer(metaclass=SingletonMeta):
    def __init__(self, resource_file):
        self.__parser = configparser.ConfigParser()
        self.__parser.read(resource_file)

    @property
    def facenet_model(self):
        return self.__parser['MODELS']['FACENET_MODEL']

    @property
    def recognition_system_configuration_file(self):
        return self.__parser['RECOGNITION_SYSTEM']['CONFIGURATION_FILE']

    @property
    def detection_system_configuration_file(self):
        return self.__parser['DETECTION_SYSTEM']['CONFIGURATION_FILE']
