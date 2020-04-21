from RecognitionModel import RecognitionModel
from os import path, listdir, mkdir
import re


class ModelVersioning:
    __instances = {}

    def __init__(self):
        raise NotImplementedError("Cannot directly create instance. Use factory method instead!")

    def __new__(cls, *args, **kwargs):
        model_directory = args[0]
        if model_directory in cls.__instances:
            return cls.__instances[model_directory]
        instance = object.__new__(cls)
        instance.__models_directory = model_directory
        instance.__model_version_regex = '.+-v([0-9]+)'
        instance.__model_filename_regex = '{}-v[0-9]+'
        instance.__version_suffix = '-v{}'
        cls.__instances[model_directory] = instance
        return instance

    @classmethod
    def get_versioning(cls, models_directory: str):
        return cls.__new__(cls, models_directory)

    def publish_model(self, model_basename: str, model: RecognitionModel) -> None:
        model_dir, models = self.__get_models(model_basename, create_notfound_dir=True)

        version = max(models, key=lambda model: model[1])[1] + 1 if len(models) > 0 else 1
        RecognitionModel.save_model_as_binary(model, path.join(model_dir, self.__version_suffix.format(version)))

    def get_model(self, model_basename: str, version: str = 'latest') -> RecognitionModel:
        model_dir, models = self.__get_models(model_basename)

        if version == 'latest':
            model_to_load = max(models, key=lambda model: model[1])[0]
        elif version == 'earliest':
            model_to_load = min(models, key=lambda model: model[1])[0]
        else:
            raise ValueError("Unsupported versioning type")

        return RecognitionModel.load_model_from_binary(path.join(model_dir, model_to_load))

    def __get_models(self, model_basename: str, create_notfound_dir: bool = False):
        model_dir = path.join(self.__models_directory, model_basename)

        if not path.isdir(model_dir):
            if create_notfound_dir:
                mkdir(model_dir)
            else:
                raise LookupError("No such model being versioned")

        return model_dir, [(model, int(re.search(self.__model_version_regex, model).group(1))) for model in listdir(model_dir)
                           if re.match(self.__model_filename_regex.format(model_basename), model)]
