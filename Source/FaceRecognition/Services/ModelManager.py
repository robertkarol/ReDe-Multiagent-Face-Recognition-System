from RecognitionModel import RecognitionModel
from Services.DirectoryContentVersioner import DirectoryContentVersioner
from Utils.Singleton import SingletonPerKey
from os import path, mkdir


class ModelManager(SingletonPerKey):

    def __new__(cls, *args, **kwargs):
        model_directory = args[0]
        instance = super().__new__(cls, model_directory)
        instance.__models_directory = model_directory
        instance.__model_versioner = DirectoryContentVersioner()
        return instance

    @classmethod
    def get_manager(cls, models_directory: str):
        return cls.__new__(cls, models_directory)

    def publish_model(self, model_basename: str, model: RecognitionModel) -> None:
        model_dir = self.__get_model_directory(model_basename)

        version = self.__model_versioner.get_latest(model_dir, model_basename)[1] + 1
        RecognitionModel.save_model_as_binary(model, path.join(model_dir, self.__model_versioner.get_versioned_name(
            model_basename, version)))

    def get_model(self, model_basename: str, version: str = 'latest') -> RecognitionModel:
        model_dir = self.__get_model_directory(model_basename)

        if not path.isdir(model_dir):
            raise LookupError(f"No such {model_dir} model being versioned")

        if version == 'latest':
            model_to_load = self.__model_versioner.get_latest(model_dir, model_basename)[0]
        elif version == 'earliest':
            model_to_load = self.__model_versioner.get_earliest(model_dir, model_basename)[0]
        else:
            raise ValueError(f"{version} is an unsupported versioning type")

        return RecognitionModel.load_model_from_binary(path.join(model_dir, model_to_load))

    def __get_model_directory(self, model_basename):
        return path.join(self.__models_directory, model_basename)

    def __get_models(self, model_basename: str, create_notfound_dir: bool = False):
        model_dir = path.join(self.__models_directory, model_basename)

        if not path.isdir(model_dir):
            if create_notfound_dir:
                mkdir(model_dir)
            else:
                raise LookupError("No such model being versioned")

        return model_dir, self.__model_versioner.get_versioned_content(model_dir, model_basename)
