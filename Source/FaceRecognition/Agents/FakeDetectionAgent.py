from Agents.SystemAgent import SystemAgent
from Domain.Connection import Connection
from Domain.RecognitionRequest import RecognitionRequest
from Domain.RecognitionResponse import RecognitionResponse
from Utils.DatasetHelpers import DatasetHelpers
from concurrent.futures.thread import ThreadPoolExecutor
from spade.behaviour import PeriodicBehaviour, CyclicBehaviour
import asyncio
import codecs
import io
import random


class FakeDetectionAgent(SystemAgent):
    class FakeDetectionBehavior(PeriodicBehaviour):
        def __init__(self, outer_ref, period):
            super().__init__(period=period)
            self.__outer_ref: FakeDetectionAgent = outer_ref
            try:
                self.__face_images = DatasetHelpers.load_images(self.__outer_ref.data_directory)
            except FileNotFoundError as error:
                self.__outer_ref.log(f"Error loading files: {error}", "critical")
                self.__outer_ref.stop()

        async def on_start(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting fake detection. . .", "info")

        async def run(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} sending face image. . .", "info")
            image = self.__get_random_face_image()
            request = RecognitionRequest(str(self.__outer_ref.jid), self.__outer_ref.agent_location,
                                         self.__image_to_base64(image), False, base64encoded=True)
            data = str.encode(RecognitionRequest.serialize(request))
            try:
                await self.__outer_ref._connection.write_data(data)
            except ConnectionError as error:
                self.__outer_ref.log(f"Error reading from connection: {error}", "critical")
                self.__outer_ref.stop()
            self.__outer_ref.log(f"{self.__outer_ref.jid} done sending face image . . .", "info")

        async def on_end(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} ending fake detection. . .", "info")

        def __image_to_base64(self, image):
            byte = io.BytesIO()
            image.save(byte, 'JPEG')
            return codecs.encode(byte.getvalue(), 'base64').decode()

        def __get_random_face_image(self):
            return random.choice(self.__face_images)

    class ResponseReceiverBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: FakeDetectionAgent = outer_ref

        async def on_start(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting responses receiver. . .", "info")

        async def run(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} waiting for response. . .", "info")
            try:
                data = await self.__outer_ref._connection.read_data()
                if not data:
                    self.kill()
                print(f'Received: {RecognitionResponse.deserialize(data)!r}')
            except ConnectionError as error:
                self.__outer_ref.log(f"Error reading from connection: {error}", "critical")
                self.__outer_ref.stop()
            self.__outer_ref.log(f"{self.__outer_ref.jid} done processing response . . .", "info")

        async def on_end(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} ending responses receiver. . .", "info")

    def __init__(self, jid: str, password: str, agent_location: str, data_directory: str, executor: ThreadPoolExecutor,
                 recognition_system_ip: str, recognition_system_port: int, detection_interval: int = 1,
                 message_checking_interval: int = 5, verify_security: bool = False):
        super().__init__(jid, password, executor, verify_security, message_checking_interval)
        self.__agent_location = agent_location
        self.__data_directory = data_directory
        self.__recognition_system_ip = recognition_system_ip
        self.__recognition_system_port = recognition_system_port
        self._connection: Connection
        self.__detection_interval = detection_interval

    @property
    def agent_location(self):
        return self.__agent_location

    @property
    def data_directory(self):
        return self.__data_directory

    @property
    def detection_interval(self):
        return self.__detection_interval

    @property
    def recognition_system_ip(self):
        return self.__recognition_system_ip

    @property
    def recognition_system_port(self):
        return self.__recognition_system_port

    async def setup(self):
        await super().setup()
        reader_stream, writer_stream = \
            await asyncio.open_connection(self.__recognition_system_ip, self.__recognition_system_port)
        self._connection = Connection(str(self.jid), reader_stream, writer_stream)
        detection_behavior = self.FakeDetectionBehavior(self, self.detection_interval)
        receiver_behavior = self.ResponseReceiverBehavior(self)
        self.add_behaviour(detection_behavior)
        self.add_behaviour(receiver_behavior)

    def stop(self):
        self._connection.close()
        return super().stop()
