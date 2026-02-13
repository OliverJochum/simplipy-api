from llama_cpp import Llama
from model_services.model_service import ModelService
from constants import (
    GENERATE_SIMPLIFIED_TEXT_PROMPT,
    SENTENCE_SIMPLIFICATION_PROMPT,
    SENTENCE_SUGGESTION_PROMPT,
)

class CustomLlamaService(ModelService):

    def prompt(self, usr_prompt: str, sys_prompt: str|None = None) -> str:
        llm = Llama.from_pretrained(
            repo_id="tschomacker/lora_Llama-3.1-8B-Instruct-bnb-4bit_gguf",
            filename="unsloth.Q4_K_M.gguf",
            n_ctx=8192,
        )
        messages = []

        if sys_prompt is None:
            messages = [{"role": "user","content": usr_prompt}]
        else:
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user","content": usr_prompt}
                    ]
        response = llm.create_chat_completion(messages=messages)
        return response["choices"][0]["message"]["content"]

    def generate_simplified_text(self, input_text: str) -> str:
        return self.prompt(input_text)
    
    def generate_sentence_simplifications(self, input_text: str) -> list[str]:
        return self.prompt(input_text, SENTENCE_SIMPLIFICATION_PROMPT)
    
    def generate_sentence_suggestions(self, input_text: str) -> list[str]:
        return self.prompt(input_text, SENTENCE_SUGGESTION_PROMPT)