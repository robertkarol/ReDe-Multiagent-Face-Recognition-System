import configparser


class ResourceLocalizerMeta(type):
    _instance = None

    def __call__(self):
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance


class ResourceLocalizer(metaclass=ResourceLocalizerMeta):
    def __init__(self):
        self.__parser = configparser.ConfigParser()
        self.__parser.read("resources.ini")

    @property
    def FaceNetModel(self):
        return self.__parser['MODELS']['FACENET_MODEL']

    @property
    def SystemConfigurationFile(self):
        return self.__parser['SYSTEM']['CONFIGURATION_FILE']
