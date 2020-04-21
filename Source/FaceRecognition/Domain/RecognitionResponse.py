import pickle
from enum import Enum


class RecognitionOutcome(Enum):
    NOT_RECOGNIZED = 1
    RECOGNIZED = 2
    UNCERTAIN = 3
    UNKNOWN = 4


# TODO: Support cross platform serialization
class RecognitionResponse:

    def __init__(self, recognized_class, probability: float, outcome: RecognitionOutcome = RecognitionOutcome.UNKNOWN):
        self.__recognized_class = recognized_class
        self.__probability = probability
        self.__outcome = outcome

    @property
    def recognized_class(self):
        return self.__recognized_class

    @property
    def recognized_class_probability(self):
        return self.__probability

    @property
    def recognition_outcome(self):
        return self.__outcome

    @classmethod
    def deserialize(cls, bytes):
        response = pickle.loads(bytes)
        return response

    @staticmethod
    def serialize(response):
        return pickle.dumps(response)
