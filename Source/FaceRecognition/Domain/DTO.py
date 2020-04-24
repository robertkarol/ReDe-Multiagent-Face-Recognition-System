from dataclasses import dataclass


@dataclass
class RecognitionAgentDTO:
    jid: str
    model_directory: str
    model_basename: str
