import json


class Serializable:
    @classmethod
    def serialize(cls, instance):
        return json.dumps(instance.__dict__)

    @classmethod
    def deserialize(cls, serialization):
        if isinstance(serialization, bytes):
            serialization = serialization.decode()
        instance = json.loads(serialization)
        return instance
