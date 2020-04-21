import pickle


# TODO: Support cross platform serialization
class RecognitionRequest:

    def __init__(self, agent_name: str, agent_location: str, face_image, generate_outcome: bool = False):
        self.__agent_name = agent_name
        self.__agent_location = agent_location
        self.__face_image = face_image
        self.__generate_outcome = generate_outcome

    @property
    def face_image(self):
        return self.__face_image

    @property
    def detection_agent(self):
        return self.__agent_name

    @property
    def detection_location(self):
        return self.__agent_location

    @property
    def generate_outcome(self):
        return self.__generate_outcome

    @classmethod
    def deserialize(cls, bytes):
        request = pickle.loads(bytes)
        return request

    @staticmethod
    def serialize(request):
        return pickle.dumps(request)
