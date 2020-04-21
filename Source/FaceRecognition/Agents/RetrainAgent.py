from Services.ModelManager import ModelManager
from concurrent.futures.thread import ThreadPoolExecutor
from spade.agent import Agent
from spade.behaviour import PeriodicBehaviour
import asyncio

from Services.NewIdentitiesManager import NewIdentitiesManager


class RetrainAgent(Agent):
    class ModelRetrainBehavior(PeriodicBehaviour):
        def __init__(self, outer_ref, period):
            super().__init__(period)
            self.__outer_ref: RetrainAgent = outer_ref
            self.__model_manager = ModelManager.get_versioning(self.__outer_ref.model_directory)
            self.__new_identities_manager = NewIdentitiesManager(list(self.__outer_ref.location_model_dict.keys()),
                                                                 self.__outer_ref.data_directory)
            self.__model = None

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} starting retrain. . .")
            for location in self.__outer_ref.location_model_dict:
                new_ident_cnt, new_ident_path = self.__new_identities_manager.get_new_identities_dataset_path(location)
                if len(new_ident_cnt) == 0:
                    continue
                await self.__retrain(self.__outer_ref.location_model_dict[location], new_ident_path)
            print(f"{self.__outer_ref.jid} done retrain . . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")

        async def __retrain(self, modeldir_modelbasename_pair: (str, str), dataset_path):
            pass

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
        retrain_behav = self.ModelRetrainBehavior(self, period=self.period)
        self.add_behaviour(retrain_behav)
