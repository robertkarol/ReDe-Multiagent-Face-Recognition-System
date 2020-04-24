from Agents.RecognitionAgent import RecognitionAgent
from Domain.DTO import RecognitionAgentDTO
from Utils.Singleton import SingletonMeta


class RecognitionLocationsManager(metaclass=SingletonMeta):
    def __init__(self):
        self.__locations = {}

    def register_recognition_locations(self, locations):
        for location in locations:
            if location not in self.__locations:
                self.__locations[location] = []

    def register_recognition_location(self, location):
        self.register_recognition_locations([location])

    def unregister_recognition_location(self, location):
        del self.__locations[location]

    def add_recognition_agents(self, location, agents: list):
        self.__locations[location].extend([RecognitionAgentDTO(str(agent.jid), agent.model_directory,
                                                               agent.model_basename) for agent in agents])

    def add_recognition_agent(self, location, agent: RecognitionAgent):
        self.add_recognition_agents(location, [agent])

    def register_and_set_from_dictionary(self, location_agents_dict):
        for location in location_agents_dict:
            self.register_recognition_location(location)
            self.add_recognition_agents(location, location_agents_dict[location])

    def get_recognition_agents(self, location):
        return self.__locations[location]

    def get_recognition_locations(self):
        return self.__locations.keys()
