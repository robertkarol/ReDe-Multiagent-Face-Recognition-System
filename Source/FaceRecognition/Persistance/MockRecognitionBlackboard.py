from Domain.DTO import RecognitionRequestDTO
from Domain.RecognitionRequest import RecognitionRequest
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.MockConnection import MockConnection
from Utils.DatasetHelpers import DatasetHelpers
from asyncinit import asyncinit
import codecs
import io


@asyncinit
class MockRecognitionBlackboard(RecognitionBlackboard):
    async def __init__(self, agent_locations):
        super().__init__(agent_locations)
        locations_count = len(agent_locations)
        images_to_predict = []
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/robi'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/mindy_kaling'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/madonna'))
        # images_to_predict is 16; we make it 320 for each agent => 1600 total
        images_to_predict = images_to_predict * 20
        agents = [images_to_predict[:]] * locations_count
        for i, agent in enumerate(agents):
            requests = self.__create_fake_request(agent, agent_locations[i])
            await super().publish_recognition_requests(agent_locations[i], requests)

    def __create_fake_request(self, agent_images, location):
        requests = []
        for i in range(len(agent_images)):
            image = agent_images[i]
            byte = io.BytesIO()
            image.save(byte, 'JPEG')
            request = RecognitionRequest("aa", location, codecs.encode(byte.getvalue(), 'base64').decode(), False,
                                   base64encoded=True)
            requests.append(RecognitionRequestDTO(MockConnection().fake_conn_id, request))
        return requests
