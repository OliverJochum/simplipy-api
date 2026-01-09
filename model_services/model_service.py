from abc import ABC, abstractmethod

class ModelService(ABC):
    @abstractmethod
    def generate_simplified_text(self, input_text: str) -> str:
        pass