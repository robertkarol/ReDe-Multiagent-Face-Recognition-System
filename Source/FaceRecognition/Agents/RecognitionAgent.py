from Persistance.RecognitionBlackboard import RecognitionBlackboard
from RecognitionModel import RecognitionModel
from Services.ModelVersioning import ModelVersioning
from concurrent.futures.thread import ThreadPoolExecutor
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
import asyncio


class RecognitionAgent(Agent):
    class MonitoringRecognitionRequestsBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: RecognitionAgent = outer_ref
            self.__model_versioning = ModelVersioning.get_versioning(self.__outer_ref.model_directory)
            self.__model = None

        async def on_start(self):
            self.__model = await self.__load_model()
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} polling. . .")
            data = self.__outer_ref.blackboard.get_recognition_requests(self.__outer_ref.location_to_serve,
                                                                        self.__outer_ref.processing_batch_size)
            if len(data) == 0:
                await asyncio.sleep(self.__outer_ref.polling_interval)
                return
            print(f"{self.__outer_ref.jid} starting resolving. . .")
            requesting_agents, faces = self.__unwrap_requests(data)
            # TODO: Add outcome gerneration if requested so
            results = await self.__outer_ref.loop.run_in_executor(None,
                                                                  lambda: self.__model.predict_from_faces_images(faces))
            self.__outer_ref.blackboard.publish_recognition_results(self.__wrap_results(requesting_agents, results))
            print(f"{self.__outer_ref.jid} done resolving . . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

        async def __load_model(self):
            print(f"{self.__outer_ref.jid} loading model . . .")
            model = await self.__outer_ref.loop.run_in_executor(None, lambda: self.__model_versioning.get_model(
                self.__outer_ref.model_basename))
            print(f"{self.__outer_ref.jid} done loading model . . .")
            return model

        def __unwrap_requests(self, raw_data):
            agents = []
            faces = []
            for i, req in enumerate(raw_data):
                agents.append(req[0])
                faces.append(req[1].face_image)
            return agents, faces

        def __wrap_results(self, agents, raw_results):
            results = []
            for i, res in enumerate(raw_results):
                results.append((agents[i], res))
            return results

    def __init__(self, jid: str, password: str, blackboard: RecognitionBlackboard, location_to_serve: str,
                 model_directory: str, model_basename: str, executor: ThreadPoolExecutor,
                 processing_batch_size: int = 5, polling_interval: float = 1, verify_security: bool = False):
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
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print(f"Agent {self.jid} starting . . .")
        b = self.MonitoringRecognitionRequestsBehavior(self)
        self.add_behaviour(b)
