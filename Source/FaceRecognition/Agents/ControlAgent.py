from Domain.RecognitionRequest import RecognitionRequest
from Domain.RecognitionResponse import RecognitionResponse, RecognitionOutcome
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.InterfaceServer import InterfaceServer
from concurrent.futures.thread import ThreadPoolExecutor
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
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
            results = self.__outer_ref.blackboard.get_recognition_results(self.__outer_ref.processing_batch_size)
            if len(results) == 0:
                await asyncio.sleep(self.__outer_ref.polling_interval)
            else:
                print(f"{self.__outer_ref.jid} starting resolving results. . .")
                await self.__outer_ref.loop.run_in_executor(
                    None, lambda: self.__outer_ref.interface_server.enqueue_responses(self.__wrap_results(results)))
                print(f"{self.__outer_ref.jid} done resolving results. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring results. . .")

        def __wrap_results(self, results):
            responses = [(result[0][0], self.__get_response(result)) for result in results]
            return responses

        def __get_response(self, result):
            gen_out, rec_class, proba = result[0][1], result[1][0], result[1][1]
            if gen_out:
                if proba < self.__outer_ref.unrecognized_threshold:
                    outcome = RecognitionOutcome.NOT_RECOGNIZED
                elif proba >= self.__outer_ref.recognized_threshold:
                    outcome = RecognitionOutcome.RECOGNIZED
                else:
                    outcome = RecognitionOutcome.UNCERTAIN
            else:
                outcome = RecognitionOutcome.UNKNOWN
            return RecognitionResponse.serialize(RecognitionResponse(str(rec_class), proba, outcome.name))

    class RecognitionRequestsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: ControlAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring requests. . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} waiting for requests. . .")
            requests = await self.__outer_ref.loop.run_in_executor(
                None, lambda: self.__outer_ref.interface_server.dequeue_requests(
                    self.__outer_ref.processing_batch_size))
            if len(requests) == 0:
                await asyncio.sleep(self.__outer_ref.polling_interval)
            else:
                print(f"{self.__outer_ref.jid} starting resolving requests. . .")
                for request in requests:
                    recognition_request: RecognitionRequest = RecognitionRequest.deserialize(request[1])
                    self.__outer_ref.blackboard.publish_recognition_request(
                        recognition_request.detection_location, (request[0], recognition_request))
                print(f"{self.__outer_ref.jid} done resolving requests. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring requests. . .")

    def __init__(self, jid: str, password: str, blackboard: RecognitionBlackboard, interface_server: InterfaceServer,
                 executor: ThreadPoolExecutor, processing_batch_size: int = 10, polling_interval: float = 1,
                 recognized_threshold: float = 0.85, unrecognized_threshold: float = 0.65,
                 verify_security: bool = False):
        self.jid = jid
        self.blackboard = blackboard
        self.password = password
        self.interface_server = interface_server
        self.loop = asyncio.get_event_loop()
        self.loop.set_default_executor(executor)
        self.processing_batch_size = processing_batch_size
        self.polling_interval = polling_interval
        self.recognized_threshold = recognized_threshold
        self.unrecognized_threshold = unrecognized_threshold
        super().__init__(jid, password, verify_security)

    async def setup(self):
        print(f"Agent {self.jid} starting . . .")
        res_beh = self.RecognitionResultsMonitoringBehavior(self)
        req_beh = self.RecognitionRequestsMonitoringBehavior(self)
        self.add_behaviour(res_beh)
        self.add_behaviour(req_beh)
