from Agents.ControlAgent import ControlAgent
from Agents.RecognitionAgent import RecognitionAgent
from Persistance.MockRecognitionBlackboard import MockRecognitionBlackboard
from Server.InterfaceServer import InterfaceServer
from concurrent import futures
import multiprocessing


if __name__ == "__main__":
    executor = futures.ThreadPoolExecutor(max_workers=8)
    responses = multiprocessing.Queue()
    requests = multiprocessing.Queue()
    server = InterfaceServer(requests, responses, ip='127.0.0.1', port=8888)
    blackboard = MockRecognitionBlackboard()
    recog_ag_count = 5

    ctrl = ControlAgent("robertkarol-ctrl1@404.city", "MeMeS-4TheWin", blackboard, server, executor)
    ag1 = RecognitionAgent("robertkarol-rec1@404.city", "MeMeS-4TheWin", blackboard, '1', 'locals', 'partial2', executor)
    ag2 = RecognitionAgent("robertkarol-rec2@404.city", "MeMeS-4TheWin", blackboard, '2', 'locals', 'partial2', executor)
    ag3 = RecognitionAgent("robertkarol-rec3@404.city", "MeMeS-4TheWin", blackboard, '3', 'locals', 'partial3', executor)
    ag4 = RecognitionAgent("robertkarol-rec4@404.city", "MeMeS-4TheWin", blackboard, '4', 'locals', 'partial4', executor)
    ag5 = RecognitionAgent("robertkarol-rec5@404.city", "MeMeS-4TheWin", blackboard, '5', 'locals', 'partial5', executor)

    server.start()
    ctrl.start()
    ag1.start()
    ag2.start()
    ag3.start()
    ag4.start()
    ag5.start()
