from Agents.SystemAgent import SystemAgent
from Domain.DTO import RecognitionResultDTO
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from PIL import Image, UnidentifiedImageError
from Services.ModelManager import ModelManager
from concurrent.futures.thread import ThreadPoolExecutor
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import asyncio
import codecs
import io


class RecognitionAgent(SystemAgent):
    @staticmethod
    def get_agent_clone(agent):
        return RecognitionAgent(str(agent.jid), agent.password, agent.blackboard, agent.location_to_serve,
                                agent.model_directory, agent.model_basename, None,
                                agent.processing_batch_size, agent.polling_interval, agent.message_checking_interval,
                                agent.verify_security)

    class MonitoringRecognitionRequestsBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: RecognitionAgent = outer_ref

        async def on_start(self):
            try:
                self.__outer_ref.model = await self.__outer_ref.load_model()
            except Exception as err:
                self.__outer_ref.log(f"Fatal exception loading model: {err}", "critical")
                self.kill()
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting the monitoring . . .", "info")

        async def run(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} polling. . .", "info")
            data = await self.__outer_ref.blackboard.get_recognition_requests(
                self.__outer_ref.location_to_serve, self.__outer_ref.processing_batch_size)
            if len(data) == 0:
                await asyncio.sleep(self.__outer_ref.polling_interval)
                return
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting resolving. . .", "info")
            requesting_agents, faces = self.__unwrap_requests(data)
            try:
                results = await self.__outer_ref.loop.run_in_executor(
                    None, lambda: self.__outer_ref.model.predict_from_faces_images(faces))
                await self.__outer_ref.blackboard.publish_recognition_results(
                    self.__wrap_results(requesting_agents, results))
            except Exception as err:
                self.__outer_ref.log(f"Exception encountered on model prediction: {err}", "error")
            self.__outer_ref.log(f"{self.__outer_ref.jid} done resolving . . .", "info")

        async def on_end(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} ending the monitoring . . .", "info")

        def __unwrap_requests(self, requests):
            agents = []
            faces = []
            for request in requests:
                conn, request = request.connection_id, request.recognition_request
                agents.append((conn, request.generate_outcome))
                serialized_image = request.face_image
                if request.base64encoded:
                    serialized_image = codecs.decode(serialized_image.encode(), 'base64')
                else:
                    serialized_image = bytes.fromhex(serialized_image)
                serialized_image = io.BytesIO(serialized_image)
                serialized_image.seek(0)
                try:
                    faces.append(Image.open(serialized_image))
                except UnidentifiedImageError as error:
                    self.__outer_ref.log(f"Error opening image: {error}", "error")
            return agents, faces

        def __wrap_results(self, agents, raw_results):
            results = []
            for i, res in enumerate(raw_results):
                results.append(RecognitionResultDTO(agents[i][0], agents[i][1], res))
            return results

    def __init__(self, jid: str, password: str, blackboard: RecognitionBlackboard, location_to_serve: str,
                 model_directory: str, model_basename: str, executor: ThreadPoolExecutor,
                 processing_batch_size: int = 5, polling_interval: float = 1,
                 message_checking_interval: int = 5, verify_security: bool = False):
        super().__init__(jid, password, executor, verify_security, message_checking_interval)
        self.__blackboard = blackboard
        self.__location_to_serve = location_to_serve
        self.__model_directory = model_directory
        self.__model_basename = model_basename
        self.__processing_batch_size = processing_batch_size
        self.__polling_interval = polling_interval
        self.__model_manager = ModelManager.get_manager(self.model_directory)
        self.__model = None

    @property
    def blackboard(self):
        return self.__blackboard

    @property
    def location_to_serve(self):
        return self.__location_to_serve

    @property
    def model_directory(self):
        return self.__model_directory

    @property
    def model_basename(self):
        return self.__model_basename

    @property
    def processing_batch_size(self):
        return self.__processing_batch_size

    @property
    def polling_interval(self):
        return self.__polling_interval

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, value):
        self.__model = value

    async def setup(self):
        await super().setup()
        rec_behavior = self.MonitoringRecognitionRequestsBehavior(self)
        self.add_behaviour(rec_behavior)

    async def load_model(self):
        self.log(f"{self.jid} loading model {self.model_basename} . . .", "info")
        model = await self.loop.run_in_executor(None, lambda: self.__model_manager.get_model(self.model_basename))
        self.log(f"{self.jid} done loading model {self.model_basename} . . .", "info")
        return model

    async def _process_message(self, message: Message):
        await super()._process_message(message)
        if message.metadata['type'] == 'new_model_available':
            self.model = await self.load_model()
