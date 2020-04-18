from DatasetHelpers import DatasetHelpers
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.MockConnection import MockConnection


class MockRecognitionBlackboard(RecognitionBlackboard):
    def __init__(self):
        super().__init__([1, 2, 3, 4, 5])
        images_to_predict = []
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/robi'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/mindy_kaling'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/madonna'))
        # images_to_predict is 16; we make it 320 for each agent => 1600 total
        agent1 = images_to_predict[:] * 20
        agent2 = images_to_predict[:] * 20
        agent3 = images_to_predict[:] * 20
        agent4 = images_to_predict[:] * 20
        agent5 = images_to_predict[:] * 20
        self.__fake_add_agent_to_respond(agent1)
        self.__fake_add_agent_to_respond(agent2)
        self.__fake_add_agent_to_respond(agent3)
        self.__fake_add_agent_to_respond(agent4)
        self.__fake_add_agent_to_respond(agent5)
        super().publish_recognition_requests(1, agent1)
        super().publish_recognition_requests(2, agent2)
        super().publish_recognition_requests(3, agent3)
        super().publish_recognition_requests(4, agent4)
        super().publish_recognition_requests(5, agent5)


    def __fake_add_agent_to_respond(self, agent_images):
        for i in range(len(agent_images)):
            agent_images[i] = (MockConnection(), agent_images[i])
