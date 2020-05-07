from dataclasses import dataclass
from typing import Any

from Domain.RecognitionRequest import RecognitionRequest


@dataclass
class RecognitionAgentDTO:
    jid: str
    model_directory: str
    model_basename: str


@dataclass
class RecognitionRequestDTO:
    connection_id: str
    recognition_request: Any


@dataclass
class RecognitionResultDTO:
    connection_id: str
    generate_outcome: bool
    recognition_result: Any
