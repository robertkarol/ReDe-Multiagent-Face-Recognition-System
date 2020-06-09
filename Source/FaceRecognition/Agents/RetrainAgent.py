from Agents.SystemAgent import SystemAgent
from Services.ModelManager import ModelManager
from Services.NewIdentitiesManager import NewIdentitiesManager
from Services.RecognitionLocationsManager import RecognitionLocationsManager
from concurrent.futures.thread import ThreadPoolExecutor
from spade.behaviour import PeriodicBehaviour
from spade.message import Message


class RetrainAgent(SystemAgent):
    class ModelRetrainBehavior(PeriodicBehaviour):
        def __init__(self, outer_ref, period):
            super().__init__(period=period)
            self.__outer_ref: RetrainAgent = outer_ref

        async def on_start(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting the monitoring . . .", "info")

        async def run(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting retrain. . .", "info")
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
            self.__outer_ref.log(f"{self.__outer_ref.jid} done retrain . . .", "info")

        async def on_end(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} ending the monitoring . . .", "info")

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
                 period: int = 120, message_checking_interval: int = 5, verify_security: bool = False):
        super().__init__(jid, password, executor, verify_security, message_checking_interval)
        self.__data_directory = data_directory
        self.__recognition_locations_manager = recognition_locations_manager
        self.__period = period
        self.__new_identities_manager: NewIdentitiesManager = NewIdentitiesManager.get_manager(data_directory)

    @property
    def data_directory(self):
        return self.__data_directory

    @property
    def period(self):
        return self.__period

    @property
    def recognition_locations_manager(self):
        return self.__recognition_locations_manager

    @property
    def new_identities_manager(self):
        return self.__new_identities_manager

    async def setup(self):
        await super().setup()
        retrain_behavior = self.ModelRetrainBehavior(self, self.period)
        self.add_behaviour(retrain_behavior)
