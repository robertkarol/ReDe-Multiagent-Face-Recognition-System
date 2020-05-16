from Services.ModelManager import ModelManager
from Services.NewIdentitiesManager import NewIdentitiesManager
from Services.RecognitionLocationsManager import RecognitionLocationsManager
from concurrent.futures.thread import ThreadPoolExecutor
from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour
from spade.message import Message
import asyncio


class RetrainAgent(Agent):
    class ModelRetrainBehavior(PeriodicBehaviour):
        def __init__(self, outer_ref, period):
            super().__init__(period=period)
            self.__outer_ref: RetrainAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} starting retrain. . .")
            locations = self.__outer_ref.recognition_locations_manager.get_recognition_locations()
            for location in locations:
                try:
                    new_ident_cnt, new_ident_path = \
                        self.__outer_ref.new_identities_manager.get_newest_identities_dataset_path(
                            location)
                except LookupError:
                    new_ident_cnt = 0
                if new_ident_cnt == 0:
                    continue
                agents_for_location = \
                    self.__outer_ref.recognition_locations_manager.get_recognition_agents(location)
                updated_agent_model_pairs = await self.__outer_ref.loop.run_in_executor(
                    None, lambda: self.__retrain(agents_for_location, new_ident_path))
                await self.__send_new_model_available_message(
                    updated_agent_model_pairs)
            print(f"{self.__outer_ref.jid} done retrain . . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

        def __retrain(self, agents_for_location, dataset_path) -> list:
            updated_models = []
            for agent in agents_for_location:
                model_dir, model_basename = agent.model_directory, \
                                            agent.model_basename
                model_manager = ModelManager.get_manager(model_dir)
                model_to_update = model_manager.get_model(model_basename)
                model_to_update.retrain_from_dataset(dataset_path)
                model_manager.publish_model(model_basename, model_to_update)
                updated_models.append((agent, model_basename))
            return updated_models

        async def __send_new_model_available_message(self, updated_agent_model_pairs):
            for agent, model in updated_agent_model_pairs:
                message = Message(to=str(agent.jid), body=model, metadata={'type': 'new_model_available'})
                await self.send(message)

    def __init__(self, jid: str, password: str, data_directory: str,
                 recognition_locations_manager: RecognitionLocationsManager, executor: ThreadPoolExecutor,
                 period: int = 120, verify_security: bool = False):
        self.jid = jid
        self.password = password
        self.data_directory = data_directory
        self.recognition_locations_manager = recognition_locations_manager
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        self.period = period
        self.new_identities_manager: NewIdentitiesManager = NewIdentitiesManager.get_manager(data_directory)
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print(f"Agent {self.jid} starting . . .")
        retrain_behav = self.ModelRetrainBehavior(self, self.period)
        self.add_behaviour(retrain_behav)
