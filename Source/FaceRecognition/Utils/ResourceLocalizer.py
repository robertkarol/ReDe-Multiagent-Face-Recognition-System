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
    def system_configuration_file(self):
        return self.__parser['SYSTEM']['CONFIGURATION_FILE']
