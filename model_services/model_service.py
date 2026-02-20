from abc import ABC, abstractmethod

class ModelService(ABC):

    # Prompt model with system and user prompts, return response content
    @abstractmethod
    def prompt(self, usr_prompt: str, sys_prompt: str|None = None) -> str:
        pass

    # Generate simplified text from complex input text
    @abstractmethod
    def generate_simplified_text(self, input_text: str, glossary_string: str | None = None) -> str:
        pass
    
    # the following two methods should probably also take in a variable for number of suggestions, currently hardcoded to 3

    # Generate sentence simplifications from complex input text
    @abstractmethod
    def generate_sentence_simplifications(self, input_text: str) -> list[str]:
        pass

    # Generate alternate sentence suggestions from simple input text
    @abstractmethod
    def generate_sentence_suggestions(self, input_text: str) -> list[str]:
        pass
