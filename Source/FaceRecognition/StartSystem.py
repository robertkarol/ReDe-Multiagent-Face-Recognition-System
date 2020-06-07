from Agents.ControlAgent import ControlAgent
from Agents.RecognitionAgent import RecognitionAgent
from Agents.RetrainAgent import RetrainAgent
from Persistance.MockRecognitionBlackboard import MockRecognitionBlackboard
from Persistance.RecognitionBlackboard import RecognitionBlackboard
from ResourceLocalizer import ResourceLocalizer
from Services.RecognitionLocationsManager import RecognitionLocationsManager
from Server.InterfaceServer import InterfaceServer
from Server.RegisterIdentitiesServer import app
from concurrent import futures
from syncasync import async_to_sync
import json
import multiprocessing
import os


def start_components(components):
    for component in components:
        component.start()


@async_to_sync
async def get_blackboard(agent_locations, is_real_system):
    return RecognitionBlackboard(list(agent_locations.keys())) \
        if is_real_system else await MockRecognitionBlackboard(list(agent_locations.keys()))

if __name__ == "__main__":
    is_real_system = int(os.environ['IS_REAL_SYSTEM']) == 1
    print(f"Starting {'real' if is_real_system else 'fake'} system. . .")
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

    blackboard = get_blackboard(agent_locations, is_real_system)

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

    control_agents = [
        ControlAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     blackboard,
                     server,
                     executor,
                     recognition_locations_manager,
                     agent['agent-processing-batch-size'],
                     agent['polling-interval'],
                     agent['recognized-threshold'],
                     agent['unrecognized-threshold'],
                     agent['max-agent-load'],
                     agent['load-checking-interval'],
                     agent['message-checking-interval'])
        for agent in control_agents_config
    ]

    retrain_agents = [
        RetrainAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                     agent['agent-password'],
                     agent['data-directory'],
                     recognition_locations_manager,
                     executor,
                     agent['polling-interval'],
                     agent['message-checking-interval'])
        for agent in retrain_agents_config
    ]

    start_components([server])
    start_components(control_agents)
    start_components(recognition_agents)
    start_components(retrain_agents)
    app.run(host=config['register-server-location']['ip'], port=config['register-server-location']['port'])
