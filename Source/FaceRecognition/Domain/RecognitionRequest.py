from Utils.Serializable import Serializable
from dataclasses import dataclass


@dataclass
class RecognitionRequest(Serializable):
    agent_name: str
    detection_location: str
    face_image: str
    generate_outcome: bool
    base64encoded: bool = False

    @classmethod
    def deserialize(cls, serialization):
        instance = super().deserialize(serialization)
        if isinstance(instance, dict):
            return RecognitionRequest(**instance)
        return None
