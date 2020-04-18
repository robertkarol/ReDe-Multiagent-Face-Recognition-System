from collections import deque


class RecognitionBlackboard:
    def __init__(self, detection_source_agents):
        self.__queues = {}
        for agent in detection_source_agents:
            self.__queues[agent] = deque()
        self.__queues['results'] = deque()

    def register_detection_source_agent(self, detection_source_agent):
        self.__queues[detection_source_agent] = deque()

    def unregister_detection_source_agent(self, detection_source_agent):
        del self.__queues[detection_source_agent]

    def get_recognition_requests(self, detection_source_agent, amount=-1):
        return self.__dequeue_element(detection_source_agent, amount)

    def get_recognition_results(self, amount=-1):
        return self.__dequeue_element('results', amount)

    def publish_recognition_requests(self, detection_source_agent, recognition_requests):
        self.__enqueue_elements(detection_source_agent, recognition_requests)

    def publish_recognition_results(self, recognition_results):
        self.__enqueue_elements('results', recognition_results)

    def __dequeue_element(self, queue_id, amount):
        available_amount = len(self.__queues[queue_id])
        if amount == -1 or available_amount < amount:
            amount = available_amount
        elements = [self.__queues[queue_id].popleft() for _ in range(amount)]
        return elements

    def __enqueue_elements(self, queue_id, elements):
        self.__queues[queue_id].extend(elements)
