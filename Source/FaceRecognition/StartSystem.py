from Agents.ControlAgent import ControlAgent
from Agents.RecognitionAgent import RecognitionAgent
from Agents.RetrainAgent import RetrainAgent
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from ResourceLocalizer import ResourceLocalizer
from Services.RecognitionLocationsManager import RecognitionLocationsManager
from Server.InterfaceServer import InterfaceServer
from Server.RegisterIdentitiesServer import *
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
    server = InterfaceServer(requests, responses, config['interface-server-location']['ip'],
                             config['interface-server-location']['port'], config['max-interface-server-workers-count'])

    agent_locations = {}
    for agent in recognition_agents_config:
        agent_locations[agent['location-to-serve']] = []

    blackboard = RecognitionBlackboard(list(agent_locations.keys()))

    recognition_agents = []
    for agent in recognition_agents_config:
        r = RecognitionAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                             agent['agent-password'],
                             blackboard,
                             agent['location-to-serve'],
                             agent['model-directory'],
                             agent['model-basename'],
                             executor,
                             agent['agent-processing-batch-size'],
                             agent['polling-interval'],
                             agent['message-checking-interval'])
        recognition_agents.append(r)
        agent_locations[r.location_to_serve].append(r)

    recognition_locations_manager = RecognitionLocationsManager()
    recognition_locations_manager.register_and_set_from_dictionary(agent_locations)
    recog_ag_count = len(recognition_agents)

    control_agents = [
        ControlAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     blackboard,
                     server,
                     executor,
                     agent['agent-processing-batch-size'],
                     agent['polling-interval'],
                     agent['recognized-threshold'],
                     agent['unrecognized-threshold'])
        for agent in control_agents_config
    ]

    retrain_agents = [
        RetrainAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     agent['data-directory'],
                     recognition_locations_manager,
                     executor,
                     agent['polling-interval'])
        for agent in retrain_agents_config
    ]

    start_components([server])
    start_components(control_agents)
    start_components(recognition_agents)
    start_components(retrain_agents)
    app.run(host=config['register-server-location']['ip'], port=config['register-server-location']['port'])
