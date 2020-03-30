# DISCLAIMER: This code is a bit ugly
import time
from concurrent import futures
import asyncio
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour

from DatasetHelpers import DatasetHelpers
from RecognitionModel import RecognitionModel
start_time = None # start_time - time when the first agent is done loading a model and starts resolving
end_time = None # end_time - time when the last running agent is done


class Blackboard:
    '''
    Mock blackboard. Suppose we have uniform amount of "requests" for each agent
    '''
    agent1 = []
    agent2 = []
    agent3 = []
    agent4 = []
    agent5 = []
    results = []
    def __init__(self):
        images_to_predict = []
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/robi'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/mindy_kaling'))
        images_to_predict.extend(DatasetHelpers.load_images('locals/retrain/val/madonna'))
        # images_to_predict is 16; we make it 320 for each agent => 1600 total
        self.agent1 = images_to_predict[:]*20
        self.agent2 = images_to_predict[:]*20
        self.agent3 = images_to_predict[:]*20
        self.agent4 = images_to_predict[:]*20
        self.agent5 = images_to_predict[:]*20

    def poll(self, agent, amount):
        if agent == 1:
            dt = self.agent1[-1-amount:-1]
            del self.agent1[-1-amount:-1]
            return dt
        if agent == 2:
            dt = self.agent2[-1-amount:-1]
            del self.agent2[-1-amount:-1]
            return dt
        if agent == 3:
            dt = self.agent3[-1-amount:-1]
            del self.agent3[-1-amount:-1]
            return dt
        if agent == 4:
            dt = self.agent4[-1-amount:-1]
            del self.agent4[-1-amount:-1]
            return dt
        if agent == 5:
            dt = self.agent5[-1-amount:-1]
            del self.agent5[-1-amount:-1]
            return dt

class RecognitionAgent(Agent):
    class MonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref = outer_ref
            self.__loop = asyncio.get_event_loop()
            self.__loop.set_default_executor(futures.ThreadPoolExecutor(max_workers=12))

        async def load_model(self):
            print(f"{self.__outer_ref.jid} loading model . . .")
            #self.__model = RecognitionModel.load_model_from_binary(self.__outer_ref.model)
            #loop = asyncio.get_event_loop()
            self.__model = await self.__loop.run_in_executor(None, lambda: RecognitionModel.load_model_from_binary(self.__outer_ref.model))
            print(f"{self.__outer_ref.jid} done loading model . . .")

        async def on_start(self):
            await self.load_model()
            print(f"{self.__outer_ref.jid} starting the monitoring . . .")
            global start_time
            if not start_time:
                start_time = time.time()

        async def run(self):
            print(f"{self.__outer_ref.jid} polling. . .")
            data = self.__outer_ref.blackboard.poll(self.__outer_ref.agents_to_respond, 5)
            if len(data) == 0:
                #TODO: Real behavior would be sleeping for a while before polling again
                self.kill()
                return
            print(f"{self.__outer_ref.jid} starting resolving. . .")
            #print(data)
            #await asyncio.sleep(0.5)
            result = await self.__loop.run_in_executor(None, lambda: self.__model.predict_from_faces_images(data))
            self.__outer_ref.blackboard.results.append(result)
            print(f"{self.__outer_ref.jid} done resolving . . .")
            #await asyncio.sleep(0.5)

        async def on_end(self):
            global end_time, start_time
            end_time = time.time()
            print(end_time-start_time)
            print(f"{self.__outer_ref.jid} ending the monitoring . . .")


    def __init__(self, jid, password, blackboard: Blackboard, agents_to_serve, model, verify_security=False):
        self.jid = jid
        self.model = model
        self.password = password
        self.blackboard = blackboard
        self.agents_to_respond = agents_to_serve
        #self.class_model = RecognitionModel.load_model_from_binary(model)
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print("Agent starting . . .")
        b = self.MonitoringBehavior(self)
        self.add_behaviour(b)


if __name__ == "__main__":
    blackboard = Blackboard()
    ag1 = RecognitionAgent("robertkarol-rec1@404.city", "MeMeS-4TheWin", blackboard, 1, model='partial2')
    ag2 = RecognitionAgent("robertkarol-rec2@404.city", "MeMeS-4TheWin", blackboard, 2, model='partial2')
    ag3 = RecognitionAgent("robertkarol-rec3@404.city", "MeMeS-4TheWin", blackboard, 3, model='partial3')
    ag4 = RecognitionAgent("robertkarol-rec4@404.city", "MeMeS-4TheWin", blackboard, 4, model='partial4')
    ag5 = RecognitionAgent("robertkarol-rec5@404.city", "MeMeS-4TheWin", blackboard, 5, model='partial5')
    ag1.start()
    ag2.start()
    ag3.start()
    ag4.start()
    ag5.start()