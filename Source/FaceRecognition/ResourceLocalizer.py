from Utils.Singleton import SingletonMeta
import configparser


class ResourceLocalizer(metaclass=SingletonMeta):
    def __init__(self):
        self.__parser = configparser.ConfigParser()
        self.__parser.read("resources.ini")

    @property
    def FaceNetModel(self):
        return self.__parser['MODELS']['FACENET_MODEL']

    @property
    def SystemConfigurationFile(self):
        return self.__parser['SYSTEM']['CONFIGURATION_FILE']
