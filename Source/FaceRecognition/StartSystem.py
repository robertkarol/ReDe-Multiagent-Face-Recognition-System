from Agents.ControlAgent import ControlAgent
from Agents.RecognitionAgent import RecognitionAgent
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from Server.InterfaceServer import InterfaceServer
from concurrent import futures
import json
import multiprocessing

global recog_ag_count


def start_components(components):
    for component in components:
        component.start()


if __name__ == "__main__":
    with open('configuration.json') as config_file:
        config = json.loads(config_file.read())
    recognition_agents_config = config['recognition-agents']
    control_agents_config = config['control-agents']

    executor = futures.ThreadPoolExecutor(max_workers=8)

    responses = multiprocessing.Queue()
    requests = multiprocessing.Queue()
    server = InterfaceServer(requests, responses)

    blackboard = RecognitionBlackboard([agent['location-to-serve'] for agent in recognition_agents_config])

    recognition_agents = [
        RecognitionAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                         agent['agent-password'],
                         blackboard,
                         agent['location-to-serve'],
                         f"{agent['model-directory']}/{agent['model-basename']}",
                         executor)
        for agent in recognition_agents_config
    ]
    recog_ag_count = len(recognition_agents)

    control_agents = [
        ControlAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     blackboard,
                     server,
                     executor)
        for agent in control_agents_config
    ]

    start_components([server])
    start_components(control_agents)
    start_components(recognition_agents)
