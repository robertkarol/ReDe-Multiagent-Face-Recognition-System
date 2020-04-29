import asyncio
import codecs
import io
import pickle
from DatasetHelpers import DatasetHelpers
from Domain.RecognitionRequest import RecognitionRequest
from Domain.RecognitionResponse import RecognitionResponse


async def tcp_echo_client(message):
    reader, writer = await asyncio.open_connection(
        '127.0.0.1', 8888)
    image = DatasetHelpers.load_images('locals/retrain/val/robi')[0]
    byte = io.BytesIO()
    image.save(byte, 'JPEG')
    r = RecognitionRequest("aa", "Arad", codecs.encode(byte.getvalue(), 'base64').decode(), False, base64encoded=True)
    message = RecognitionRequest.serialize(r)
    data = str.encode(message)
    writer.write(len(data).to_bytes(4, 'little'))
    writer.write(data)
    await writer.drain()
    data_len = int.from_bytes(await reader.read(4), byteorder='little')
    data = await reader.read(data_len)
    print(f'Received: {RecognitionResponse.deserialize(data)!r}')
    writer.close()

asyncio.run(tcp_echo_client('Hello World!'))
