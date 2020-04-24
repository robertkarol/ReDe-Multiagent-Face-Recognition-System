from Utils.Serializable import Serializable
from dataclasses import dataclass
from enum import Enum


class RecognitionOutcome(Enum):
    NOT_RECOGNIZED = 1
    RECOGNIZED = 2
    UNCERTAIN = 3
    UNKNOWN = 4


@dataclass
class RecognitionResponse(Serializable):
    recognized_class: str
    probability: float
    outcome: str

    @classmethod
    def deserialize(cls, serialization):
        instance = super().deserialize(serialization)
        if isinstance(instance, dict):
            return RecognitionResponse(**instance)
        return None
