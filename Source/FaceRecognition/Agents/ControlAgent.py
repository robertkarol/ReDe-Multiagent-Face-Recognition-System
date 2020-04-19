from Domain.RecognitionRequest import RecognitionRequest
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.InterfaceServer import InterfaceServer
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from typing import List, Tuple, Any
import asyncio


class ControlAgent(Agent):
    class RecognitionResultsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: ControlAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring results . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} polling for results. . .")
            data = self.__outer_ref.blackboard.get_recognition_results(self.__outer_ref.processing_batch_size)
            if len(data) == 0:
                await asyncio.sleep(1)
            else:
                print(f"{self.__outer_ref.jid} starting resolving results. . .")
                await self.__outer_ref.loop.run_in_executor(None,
                                            lambda: self.__outer_ref.interface_server.enqueue_responses(data))
                print(f"{self.__outer_ref.jid} done resolving results. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring results. . .")

    class RecognitionRequestsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: ControlAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring requests. . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} waiting for requests. . .")
            requests: List[Tuple[Any, Any]] = await self.__outer_ref.loop.run_in_executor(None,
                                                            lambda: self.__outer_ref.interface_server.dequeue_requests(
                                                                self.__outer_ref.processing_batch_size))
            if len(requests) == 0:
                await asyncio.sleep(1)
            else:
                print(f"{self.__outer_ref.jid} starting resolving requests. . .")
                for request in requests:
                    recognition_request = RecognitionRequest.deserialize_request(request[1])
                    print(recognition_request)
                    self.__outer_ref.blackboard.publish_recognition_request(recognition_request.detection_location,
                                                                    (request[0], recognition_request.face_image))
                print(f"{self.__outer_ref.jid} done resolving requests. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring requests. . .")

    def __init__(self, jid, password, blackboard: RecognitionBlackboard, interface_server: InterfaceServer,
                 executor, processing_batch_size=10, verify_security=False):
        self.jid = jid
        self.blackboard = blackboard
        self.password = password
        self.interface_server = interface_server
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        self.processing_batch_size = processing_batch_size
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print("Agent starting . . .")
        res_beh = self.RecognitionResultsMonitoringBehavior(self)
        req_beh = self.RecognitionRequestsMonitoringBehavior(self)
        self.add_behaviour(res_beh)
        self.add_behaviour(req_beh)
