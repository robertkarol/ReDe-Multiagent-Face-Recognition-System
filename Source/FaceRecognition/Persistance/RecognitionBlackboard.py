from collections import deque


class RecognitionBlackboard:
    def __init__(self, detection_sources):
        self.__queues = {}
        for source in detection_sources:
            self.__queues[source] = deque()
        self.__queues['results'] = deque()

    def register_detection_source(self, detection_source):
        self.__queues[detection_source] = deque()

    def unregister_detection_source(self, detection_source):
        del self.__queues[detection_source]

    def get_recognition_requests(self, detection_source, amount=-1):
        return self.__dequeue_element(detection_source, amount)

    def get_recognition_results(self, amount=-1):
        return self.__dequeue_element('results', amount)

    def publish_recognition_requests(self, detection_source, recognition_requests):
        self.__enqueue_elements(detection_source, recognition_requests)

    def publish_recognition_request(self, detection_source, recognition_request):
        self.publish_recognition_requests(detection_source, [recognition_request])

    def publish_recognition_results(self, recognition_results):
        self.__enqueue_elements('results', recognition_results)

    def publish_recognition_result(self, recognition_result):
        self.__enqueue_elements('results', [recognition_result])

    def __dequeue_element(self, queue_id, amount):
        available_amount = len(self.__queues[queue_id])
        if amount == -1 or available_amount < amount:
            amount = available_amount
        elements = [self.__queues[queue_id].popleft() for _ in range(amount)]
        return elements

    def __enqueue_elements(self, queue_id, elements):
        self.__queues[queue_id].extend(elements)
