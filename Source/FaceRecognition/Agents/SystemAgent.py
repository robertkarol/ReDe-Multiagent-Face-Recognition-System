from Utils.Logging import LoggingMixin
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade.message import Message


class SystemAgent(Agent, LoggingMixin):
    class MessageReceiverBehavior(CyclicBehaviour):
        def __init__(self, outer_ref):
            super().__init__()
            self.__outer_ref: SystemAgent = outer_ref

        async def on_start(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} starting the message receiver. . .", "info")

        async def run(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} checking for message. . .", "info")
            message = await self.receive(self.__outer_ref.message_checking_interval)
            if message:
                self.__outer_ref.log(f"{self.__outer_ref.jid} processing message. . .", "info")
                await self.__outer_ref._process_message(message)
                self.__outer_ref.log(f"{self.__outer_ref.jid} done processing message. . .", "info")

        async def on_end(self):
            self.__outer_ref.log(f"{self.__outer_ref.jid} ending the message receiver. . .", "info")

    async def _process_message(self, message: Message):
        pass

    @property
    def message_checking_interval(self):
        return self.__message_checking_interval

    def __init__(self, jid: str, password: str, executor, verify_security: bool = False,
                 message_checking_interval: int = 5):
        super().__init__(jid, password, verify_security)
        if executor:
            self.loop.set_default_executor(executor)
        self.__message_checking_interval = message_checking_interval

    async def setup(self):
        self.log(f"{self.jid} agent starting . . .", "info")
        if self.message_checking_interval > -1:
            msg_behavior = self.MessageReceiverBehavior(self)
            self.add_behaviour(msg_behavior)
        else:
            self.log(f"{self.jid} skipping message checking . . .", "info")

    def log(self, message, level):
        try:
            log_method = getattr(self.logger, level)
            log_method(message)
        except AttributeError:
            raise ValueError(f"{level} is not a valid logging level")
