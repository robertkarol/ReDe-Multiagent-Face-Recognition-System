from Domain.DTO import RecognitionResultDTO
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from PIL import Image
from Services.ModelManager import ModelManager
from concurrent.futures.thread import ThreadPoolExecutor
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message
import asyncio
import codecs
import io


class RecognitionAgent(Agent):
    class MonitoringRecognitionRequestsBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: RecognitionAgent = outer_ref
            self.__model = None

        async def on_start(self):
            self.__model = await self.__outer_ref.load_model()
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} polling. . .")
            data = await self.__outer_ref.blackboard.get_recognition_requests(self.__outer_ref.location_to_serve,
                                                                              self.__outer_ref.processing_batch_size)
            if len(data) == 0:
                await asyncio.sleep(self.__outer_ref.polling_interval)
                return
            print(f"{self.__outer_ref.jid} starting resolving. . .")
            requesting_agents, faces = self.__unwrap_requests(data)
            results = await self.__outer_ref.loop.run_in_executor(None,
                                                                  lambda: self.__model.predict_from_faces_images(faces))
            await self.__outer_ref.blackboard.publish_recognition_results(
                self.__wrap_results(requesting_agents, results))
            print(f"{self.__outer_ref.jid} done resolving . . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

        def __unwrap_requests(self, raw_data):
            agents = []
            faces = []
            for i, request in enumerate(raw_data):
                conn, request = request.connection_id, request.recognition_request
                agents.append((conn, request.generate_outcome))
                serialized_image = request.face_image
                if request.base64encoded:
                    serialized_image = codecs.decode(serialized_image.encode(), 'base64')
                else:
                    serialized_image = bytes.fromhex(serialized_image)
                serialized_image = io.BytesIO(serialized_image)
                serialized_image.seek(0)
                faces.append(Image.open(serialized_image))
            return agents, faces

        def __wrap_results(self, agents, raw_results):
            results = []
            for i, res in enumerate(raw_results):
                results.append(RecognitionResultDTO(agents[i][0], agents[i][1], res))
            return results

    class MessageReceiverBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: RecognitionAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting the message receiver. . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} checking for message. . .")
            message = await self.receive(self.__outer_ref.message_checking_interval)
            if message:
                print(f"{self.__outer_ref.jid} processing message. . .")
                await self.__process_message(message)
                print(f"{self.__outer_ref.jid} done processing message. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the message receiver. . .")

        async def __process_message(self, message: Message):
            if message.metadata['type'] == 'new_model_available':
                self.__model = await self.__outer_ref.load_model()

    # TODO: make fields protected for all agents
    def __init__(self, jid: str, password: str, blackboard: RecognitionBlackboard, location_to_serve: str,
                 model_directory: str, model_basename: str, executor: ThreadPoolExecutor,
                 processing_batch_size: int = 5, polling_interval: float = 1,
                 message_checking_interval: int = 5, verify_security: bool = False):
        self.jid = jid
        self.password = password
        self.blackboard = blackboard
        self.location_to_serve = location_to_serve
        self.model_directory = model_directory
        self.model_basename = model_basename
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        self.processing_batch_size = processing_batch_size
        self.polling_interval = polling_interval
        self.message_checking_interval = message_checking_interval
        self.__model_manager = ModelManager.get_manager(self.model_directory)
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print(f"Agent {self.jid} starting . . .")
        rec_behav = self.MonitoringRecognitionRequestsBehavior(self)
        msg_behav = self.MessageReceiverBehavior(self)
        self.add_behaviour(rec_behav)
        self.add_behaviour(msg_behav)

    async def load_model(self):
        print(f"{self.jid} loading model {self.model_basename} . . .")
        model = await self.loop.run_in_executor(None, lambda: self.__model_manager.get_model(self.model_basename))
        print(f"{self.jid} done loading model {self.model_basename} . . .")
        return model
