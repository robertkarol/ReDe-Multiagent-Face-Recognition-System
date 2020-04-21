from Agents.ControlAgent import ControlAgent
from Agents.RecognitionAgent import RecognitionAgent
from Agents.RetrainAgent import RetrainAgent
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from ResourceLocalizer import ResourceLocalizer
from Server.InterfaceServer import InterfaceServer
from concurrent import futures
import json
import multiprocessing

global recog_ag_count


def start_components(components):
    for component in components:
        component.start()


if __name__ == "__main__":
    resource_localizer = ResourceLocalizer()
    with open(resource_localizer.SystemConfigurationFile) as config_file:
        config = json.loads(config_file.read())
    recognition_agents_config = config['recognition-agents']
    control_agents_config = config['control-agents']
    retrain_agents_config = config['retrain-agents']

    executor = futures.ThreadPoolExecutor(max_workers=config['max-recognition-workers-count'])

    responses = multiprocessing.Queue()
    requests = multiprocessing.Queue()
    server = InterfaceServer(requests, responses, config['server-location']['ip'], config['server-location']['port'],
                             config['max-interface-server-workers-count'])

    blackboard = RecognitionBlackboard([agent['location-to-serve'] for agent in recognition_agents_config])

    recognition_agents = [
        RecognitionAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                         agent['agent-password'],
                         blackboard,
                         agent['location-to-serve'],
                         agent['model-directory'],
                         agent['model-basename'],
                         executor,
                         agent['agent-processing-batch-size'],
                         agent['polling-interval'])
        for agent in recognition_agents_config
    ]
    recog_ag_count = len(recognition_agents)

    control_agents = [
        ControlAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     blackboard,
                     server,
                     executor,
                     agent['agent-processing-batch-size'],
                     agent['polling-interval'])
        for agent in control_agents_config
    ]

    location_model_dict = {}
    for agent in recognition_agents_config:
        location_model_dict[agent['location-to-serve']] = (agent['model-directory'], agent['model-basename'])

    retrain_agents = [
        RetrainAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     agent['data-directory'],
                     location_model_dict,
                     executor,
                     agent['polling-interval'])
        for agent in retrain_agents_config
    ]

    start_components([server])
    start_components(control_agents)
    start_components(recognition_agents)
    start_components(retrain_agents)
