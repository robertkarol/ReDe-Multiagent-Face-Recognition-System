from Utils.NotifierDeque import NotifierDeque
from typing import Iterable


class RecognitionBlackboard:
    def __init__(self, detection_sources):
        self.__queues = {}
        for source in detection_sources:
            self.__queues[source] = NotifierDeque()
        self.__queues['results'] = NotifierDeque()

    async def register_detection_source(self, detection_source) -> None:
        self.__queues[detection_source] = NotifierDeque()

    async def unregister_detection_source(self, detection_source) -> None:
        del self.__queues[detection_source]

    async def get_recognition_requests(self, detection_source, amount: int = -1):
        return await self.__dequeue_element(detection_source, amount)

    async def get_recognition_results(self, amount: int = -1):
        return await self.__dequeue_element('results', amount)

    async def publish_recognition_requests(self, detection_source, recognition_requests: Iterable) -> None:
        await self.__enqueue_elements(detection_source, recognition_requests)

    async def publish_recognition_request(self, detection_source, recognition_request) -> None:
        await self.publish_recognition_requests(detection_source, [recognition_request])

    async def publish_recognition_results(self, recognition_results: Iterable) -> None:
        await self.__enqueue_elements('results', recognition_results)

    async def publish_recognition_result(self, recognition_result) -> None:
        await self.publish_recognition_results([recognition_result])

    async def __dequeue_element(self, queue_id, amount):
        available_amount = len(self.__queues[queue_id])
        if available_amount == 0:
            elements = [await self.__queues[queue_id].popleft()]
            available_amount = len(self.__queues[queue_id])
            amount -= 1
        else:
            elements = []
        amount = available_amount if amount == -1 or available_amount < amount else amount
        if amount > 0:
            elements.extend([await self.__queues[queue_id].popleft() for _ in range(amount)])
        return elements

    async def __enqueue_elements(self, queue_id, elements):
        await self.__queues[queue_id].extend(elements)

    def get_load_information(self):
        return {queue_name: len(queue) for (queue_name, queue) in self.__queues.items()}
