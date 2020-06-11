from Agents.FakeDetectionAgent import FakeDetectionAgent
from Utils.Logging import LoggingMixin
from Utils.ResourceLocalizer import ResourceLocalizer
from Utils.SystemUtils import is_real_system, start_components
import json

if __name__ == "__main__":
    real_system = is_real_system()
    logger = LoggingMixin().logger
    if real_system:
        logger.warning(f"Only fake system supported! Will start fake system instead.")
    else:
        logger.debug("Starting fake system. . .")
    resource_localizer = ResourceLocalizer("resources.ini")
    with open(resource_localizer.detection_system_configuration_file) as config_file:
        config = json.loads(config_file.read())
    fake_detection_agents_config = config['fake-detection-agents']

    fake_detection_agents = [
        FakeDetectionAgent(f"{agent['agent-name']}@{agent['agent-server']}",
                           agent['agent-password'],
                           agent['agent-location'],
                           agent['data-directory'],
                           None,
                           agent['recognition-system-ip'],
                           agent['recognition-system-port'],
                           agent['detection-interval'],
                           agent['message-checking-interval'])
        for agent in fake_detection_agents_config
    ]

    start_components(fake_detection_agents)
