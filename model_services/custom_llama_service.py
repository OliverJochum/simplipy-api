from llama_cpp import Llama
from model_services.model_service import ModelService

class CustomLlamaService(ModelService):

    def prompt(self, sys_prompt: str, usr_prompt: str) -> str:
        llm = Llama.from_pretrained(
            repo_id="tschomacker/lora_Llama-3.1-8B-Instruct-bnb-4bit_gguf",
            filename="unsloth.Q4_K_M.gguf",
            n_ctx=8192,
        )

        response = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user","content": usr_prompt}
                    ]
                )
        return response["choices"][0]["message"]["content"]

    def generate_simplified_text(self, input_text: str) -> str:
        llm = Llama.from_pretrained(
            repo_id="tschomacker/lora_Llama-3.1-8B-Instruct-bnb-4bit_gguf",
            filename="unsloth.Q4_K_M.gguf",
            n_ctx=8192,
        )

        response = llm.create_chat_completion(
            messages = [
                {"role": "user","content": input_text},
                    ]
                )
        
        return {"response": response["choices"][0]["message"]["content"]}
    
    def generate_sentence_simplifications(self, input_text: str) -> list[str]:
        system_prompt = """Input: 1 Satz in normaler deutschen Sprache. Output: 3 alternative übersetzte Sätze in Leichter Sprache. Output als Array von Strings. Striktes Output Format: ['suggestion1', 'suggestion2', 'suggestion3'] """
        
        return {"response": self.prompt(system_prompt, input_text)}
    
    def generate_sentence_suggestions(self, input_text: str) -> list[str]:
        system_prompt = """Input: 1 Satz in Leichter Sprache. Output: 3 alternative Vorschläge in Leichter Sprache. Output als Array von Strings. Striktes Output Format: ['suggestion1', 'suggestion2', 'suggestion3'] """
        
        return {"response": self.prompt(system_prompt, input_text)}
