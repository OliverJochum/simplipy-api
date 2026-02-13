from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

from model_services.model_service import ModelService
from constants import (
    THIRD_PARTY_SYSTEM_PROMPT_ROLE_PREFIX,
    THIRD_PARTY_SYSTEM_PROMPT_RULES,
    GENERATE_SIMPLIFIED_TEXT_PROMPT,
    SENTENCE_SIMPLIFICATION_PROMPT,
    SENTENCE_SUGGESTION_PROMPT,
)

class OpenAIService(ModelService):
    
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    def prompt(self, usr_prompt: str, sys_prompt: str|None = None) -> str:
        model = ChatOpenAI(model="gpt-5-mini")
        messages = []

        if sys_prompt is None:
            messages = [{"role": "user", "content": usr_prompt}]
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ]
            
        return model.invoke(messages)

    def generate_simplified_text(self, input_text: str) -> str:
        system_prompt = THIRD_PARTY_SYSTEM_PROMPT_ROLE_PREFIX + GENERATE_SIMPLIFIED_TEXT_PROMPT + THIRD_PARTY_SYSTEM_PROMPT_RULES
        return self.prompt(input_text, system_prompt).text
    
    def generate_sentence_simplifications(self, input_text: str) -> list[str]:
        system_prompt = THIRD_PARTY_SYSTEM_PROMPT_ROLE_PREFIX + SENTENCE_SIMPLIFICATION_PROMPT + THIRD_PARTY_SYSTEM_PROMPT_RULES 
        return self.prompt(input_text, system_prompt).text
    
    def generate_sentence_suggestions(self, input_text: str) -> list[str]:
        system_prompt = THIRD_PARTY_SYSTEM_PROMPT_ROLE_PREFIX + SENTENCE_SUGGESTION_PROMPT + THIRD_PARTY_SYSTEM_PROMPT_RULES
        return self.prompt(input_text, system_prompt).text