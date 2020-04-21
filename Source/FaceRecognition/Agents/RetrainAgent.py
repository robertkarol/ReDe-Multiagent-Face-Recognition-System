from Services.ModelManager import ModelManager
from Services.NewIdentitiesManager import NewIdentitiesManager
from concurrent.futures.thread import ThreadPoolExecutor
from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour, CyclicBehaviour
import asyncio


class RetrainAgent(Agent):
    class ModelRetrainBehavior(PeriodicBehaviour):
        def __init__(self, outer_ref, period):
            super().__init__(period=period)
            self.__outer_ref: RetrainAgent = outer_ref
            self.__new_identities_manager: NewIdentitiesManager = NewIdentitiesManager.get_manager(
                self.__outer_ref.data_directory, list(self.__outer_ref.location_model_dict.keys()))

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} starting retrain. . .")
            for location in self.__outer_ref.location_model_dict:
                try:
                    new_ident_cnt, new_ident_path = \
                        self.__new_identities_manager.get_newest_identities_dataset_path(location)
                except LookupError:
                    new_ident_cnt = 0
                if new_ident_cnt == 0:
                    continue
                await self.__outer_ref.loop.run_in_executor(None, lambda: self.__retrain(
                    self.__outer_ref.location_model_dict[location], new_ident_path))
            print(f"{self.__outer_ref.jid} done retrain . . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

        def __retrain(self, modeldir_modelbasename_pair: (str, str), dataset_path):
            model_dir, model_basename = modeldir_modelbasename_pair
            model_manager = ModelManager.get_manager(model_dir)
            model_to_update = model_manager.get_model(model_basename)
            model_to_update.retrain_from_dataset(dataset_path)
            model_manager.publish_model(model_basename, model_to_update)

    def __init__(self, jid: str, password: str, data_directory: str, location_model_dict: dict,
                 executor: ThreadPoolExecutor, period: int = 120, verify_security: bool = False):
        self.jid = jid
        self.password = password
        self.data_directory = data_directory
        self.location_model_dict = location_model_dict
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        self.period = period
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print(f"Agent {self.jid} starting . . .")
        retrain_behav = self.ModelRetrainBehavior(self, self.period)
        self.add_behaviour(retrain_behav)
