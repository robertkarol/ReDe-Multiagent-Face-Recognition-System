import pickle


class RecognitionRequest:

    def __init__(self, agent_name, agent_location, face_image):
        self.__agent_name = agent_name
        self.__agent_location = agent_location
        self.__face_image = face_image

    @property
    def face_image(self):
        return self.__face_image

    @property
    def detection_agent(self):
        return self.__agent_name

    @property
    def detection_location(self):
        return self.__agent_location

    @staticmethod
    def deserialize_request(bytes):
        request = pickle.loads(bytes)
        return request
