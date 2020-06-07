from Agents.RecognitionAgent import RecognitionAgent
from Agents.SystemAgent import SystemAgent
from Domain.RecognitionRequest import RecognitionRequest
from Domain.RecognitionResponse import RecognitionResponse, RecognitionOutcome
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.InterfaceServer import InterfaceServer
from Services.RecognitionLocationsManager import RecognitionLocationsManager
from concurrent.futures.thread import ThreadPoolExecutor
from json import JSONDecodeError
from spade.behaviour import CyclicBehaviour, PeriodicBehaviour
import asyncio


class ControlAgent(SystemAgent):
    class RecognitionResultsMonitoringBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: ControlAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting monitoring results . . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} polling for results. . .")
            results = await self.__outer_ref.blackboard.get_recognition_results(self.__outer_ref.processing_batch_size)
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
            for result in results:
                result.recognition_result = self.__build_response(result)
            return results

        def __build_response(self, result):
            generate_outcome, recognized_class, proba = result.generate_outcome, result.recognition_result[0], \
                                        result.recognition_result[1]
            if generate_outcome:
                if proba < self.__outer_ref.unrecognized_threshold:
                    outcome = RecognitionOutcome.NOT_RECOGNIZED
                elif proba >= self.__outer_ref.recognized_threshold:
                    outcome = RecognitionOutcome.RECOGNIZED
                else:
                    outcome = RecognitionOutcome.UNCERTAIN
            else:
                outcome = RecognitionOutcome.UNKNOWN
            return RecognitionResponse.serialize(RecognitionResponse(str(recognized_class), proba, outcome.name))

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
                    try:
                        self.__build_request(request)
                    except JSONDecodeError as error:
                        print(f"Error decoding JSON: {error}")
                        continue
                    await self.__outer_ref.blackboard.publish_recognition_request(
                        request.recognition_request.detection_location, request)
                print(f"{self.__outer_ref.jid} done resolving requests. . .")

        def __build_request(self, request):
            request.recognition_request: RecognitionRequest = \
                RecognitionRequest.deserialize(request.recognition_request)

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending monitoring requests. . .")

    class LoadManagementBehavior(PeriodicBehaviour):
        def __init__(self, outer_ref, period):
            super().__init__(period=period)
            self.__outer_ref: ControlAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting managing load. . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} checking load. . .")
            load_info = self.__outer_ref.blackboard.get_load_information()
            for location in load_info:
                if location == 'results':
                    continue
                load_factor = load_info[location] // self.__outer_ref.max_load_per_agent + 1
                agents = self.__outer_ref.recognition_locations_manager.get_recognition_agents(location)
                agents_count = len(agents)
                if agents_count > 0 and not agents[0].model:
                    continue
                running_agents_count = await self.__balance_number_of_agents(agents, load_factor)
                if running_agents_count > agents_count:
                    print(f"Added {running_agents_count - agents_count} agents for {location}")
                elif running_agents_count < agents_count:
                    print(f"Removed {agents_count - running_agents_count} agents for {location}")
            print(f"{self.__outer_ref.jid} done checking load. . .")

        async def __balance_number_of_agents(self, agents, load_factor):
            no_of_agents = len(agents)

            while load_factor > no_of_agents:
                new_agent = RecognitionAgent.get_agent_clone(agents[0])
                self.__outer_ref.recognition_locations_manager.add_recognition_agent(
                        new_agent.location_to_serve, new_agent)
                await new_agent.start()
                no_of_agents += 1

            while no_of_agents > load_factor:
                victim_agent = agents[-1]
                self.__outer_ref.recognition_locations_manager.remove_recognition_agent(
                        victim_agent.location_to_serve, victim_agent)
                await victim_agent.stop()
                no_of_agents -= 1

            return no_of_agents

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending managing load . .")

    def __init__(self, jid: str, password: str, blackboard: RecognitionBlackboard, interface_server: InterfaceServer,
                 executor: ThreadPoolExecutor, recognition_locations_manager, processing_batch_size: int = 10,
                 polling_interval: float = 1, recognized_threshold: float = 0.85, unrecognized_threshold: float = 0.65,
                 max_load_per_agent: int = 100, load_check_period: int = 5, message_checking_interval: int = 5,
                 verify_security: bool = False):
        super().__init__(jid, password, executor, verify_security, message_checking_interval)
        self.__blackboard = blackboard
        self.__interface_server = interface_server
        self.__recognition_locations_manager: RecognitionLocationsManager = recognition_locations_manager
        self.__processing_batch_size = processing_batch_size
        self.__polling_interval = polling_interval
        self.__recognized_threshold = recognized_threshold
        self.__unrecognized_threshold = unrecognized_threshold
        self.__max_load_per_agent = max_load_per_agent
        self.__load_check_period = load_check_period

    @property
    def blackboard(self):
        return self.__blackboard

    @property
    def interface_server(self):
        return self.__interface_server

    @property
    def recognition_locations_manager(self):
        return self.__recognition_locations_manager

    @property
    def processing_batch_size(self):
        return self.__processing_batch_size

    @property
    def polling_interval(self):
        return self.__polling_interval

    @property
    def recognized_threshold(self):
        return self.__recognized_threshold

    @property
    def unrecognized_threshold(self):
        return self.__unrecognized_threshold

    @property
    def max_load_per_agent(self):
        return self.__max_load_per_agent

    @property
    def load_check_period(self):
        return self.__load_check_period

    async def setup(self):
        print(f"Agent {self.jid} starting . . .")
        await super().setup()
        res_behavior = self.RecognitionResultsMonitoringBehavior(self)
        req_behavior = self.RecognitionRequestsMonitoringBehavior(self)
        self.add_behaviour(res_behavior)
        self.add_behaviour(req_behavior)
        if self.load_check_period > -1:
            load_behavior = self.LoadManagementBehavior(self, self.load_check_period)
            self.add_behaviour(load_behavior)
        else:
            print(f"Agent {self.jid} skipping load balancing . . .")
