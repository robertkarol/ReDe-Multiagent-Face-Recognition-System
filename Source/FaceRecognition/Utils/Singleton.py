class SingletonMeta(type):
    _instance = None

    def __call__(self):
        if self._instance is None:
            self._instance = super().__call__()
        return self._instance


class SingletonPerKey:
    _instances = {}

    def __init__(self):
        raise NotImplementedError("Cannot directly create instance. Use factory method instead!")

    def __new__(cls, *args, **kwargs):
        key = args[0]
        if cls in cls._instances and key in cls._instances[cls]:
            return cls._instances[cls][key]
        instance = object.__new__(cls)
        cls._instances[cls] = {}
        cls._instances[cls][key] = instance
        return instance
