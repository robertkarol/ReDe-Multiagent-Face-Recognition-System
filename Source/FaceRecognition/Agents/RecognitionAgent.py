from Persistance.RecognitionBlackboard import RecognitionBlackboard
from RecognitionModel import RecognitionModel
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
import asyncio


class RecognitionAgent(Agent):
    class MonitoringRecognitionRequestsBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref
            self.__model = None

        async def load_model(self):
            print(f"{self.__outer_ref.jid} loading model . . .")
            model = await self.__outer_ref.loop.run_in_executor(None, lambda: RecognitionModel.load_model_from_binary(
                self.__outer_ref.model))
            print(f"{self.__outer_ref.jid} done loading model . . .")
            return model

        async def on_start(self):
            self.__model = await self.load_model()
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} polling. . .")
            data = self.__outer_ref.blackboard.get_recognition_requests(self.__outer_ref.location_to_serve, 4)
            if len(data) == 0:
                await asyncio.sleep(1)
                return
            print(f"{self.__outer_ref.jid} starting resolving. . .")
            requesting_agents, faces = self.__unwrap_requests(data)
            results = await self.__outer_ref.loop.run_in_executor(None,
                                                                  lambda: self.__model.predict_from_faces_images(faces))
            self.__outer_ref.blackboard.publish_recognition_results(self.__wrap_results(requesting_agents, results))
            print(f"{self.__outer_ref.jid} done resolving . . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

        def __unwrap_requests(self, raw_data):
            agents = []
            faces = []
            for i, req in enumerate(raw_data):
                agents.append(req[0])
                faces.append(req[1])
            return agents, faces

        def __wrap_results(self, agents, raw_results):
            results = []
            for i, res in enumerate(raw_results):
                results.append((agents[i], res))
            return results

    def __init__(self, jid, password, blackboard: RecognitionBlackboard, location_to_serve, model, executor, verify_security=False):
        self.jid = jid
        self.password = password
        self.blackboard = blackboard
        self.location_to_serve = location_to_serve
        self.model = model
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print("Agent starting . . .")
        b = self.MonitoringRecognitionRequestsBehavior(self)
        self.add_behaviour(b)
