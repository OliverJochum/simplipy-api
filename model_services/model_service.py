from abc import ABC, abstractmethod

class ModelService(ABC):

    # Prompt model with system and user prompts, return response content
    @abstractmethod
    def prompt(self, sys_prompt: str, usr_prompt: str) -> str:
        pass

    # Generate simplified text from complex input text
    @abstractmethod
    def generate_simplified_text(self, input_text: str) -> str:
        pass
    
    # Generate sentence simplifications from complex input text
    @abstractmethod
    def generate_sentence_simplifications(self, input_text: str) -> list[str]:
        pass

    # Generate alternate sentence suggestions from simple input text
    @abstractmethod
    def generate_sentence_suggestions(self, input_text: str) -> list[str]:
        pass
