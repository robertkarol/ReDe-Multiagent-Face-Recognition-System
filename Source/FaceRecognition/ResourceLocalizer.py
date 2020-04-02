import configparser


class ResourceLocalizer:
    def __init__(self):
        self.__parser = configparser.ConfigParser()
        self.__parser.read("resources.ini")

    @property
    def FaceNetModel(self):
        return self.__parser['MODELS']['FACENET_MODEL']