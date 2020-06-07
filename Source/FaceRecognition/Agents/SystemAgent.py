from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message


class SystemAgent(Agent):
    class MessageReceiverBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: SystemAgent = outer_ref

        async def on_start(self):
            print(f"{self.__outer_ref.jid} starting the message receiver. . .")

        async def run(self):
            print(f"{self.__outer_ref.jid} checking for message. . .")
            message = await self.receive(self.__outer_ref.message_checking_interval)
            if message:
                print(f"{self.__outer_ref.jid} processing message. . .")
                await self.__outer_ref._process_message(message)
                print(f"{self.__outer_ref.jid} done processing message. . .")

        async def on_end(self):
            print(f"{self.__outer_ref.jid} ending the message receiver. . .")

    async def _process_message(self, message: Message):
        pass

    @property
    def message_checking_interval(self):
        return self.__message_checking_interval

    def __init__(self, jid: str, password: str, executor, verify_security: bool = False,
                 message_checking_interval: int = 5):
        super().__init__(jid, password, verify_security)
        self.loop.set_default_executor(executor)
        self.__message_checking_interval = message_checking_interval

    async def setup(self):
        if self.message_checking_interval > -1:
            msg_behavior = self.MessageReceiverBehavior(self)
            self.add_behaviour(msg_behavior)
        else:
            print(f"Agent {self.jid} skipping message checking . . .")
