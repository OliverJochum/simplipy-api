from model_services.model_service import ModelService
from model_services.openai_service import OpenAIService
from model_services.custom_llama_service import CustomLlamaService


MODELSERVICE_REGISTRY: dict[str, type[ModelService]] = {
    "openai": OpenAIService,
    "llama": CustomLlamaService,
}

def create_model_service(kind: str) -> ModelService:
    try:
        cls = MODELSERVICE_REGISTRY[kind]
        return cls()
    except KeyError:
        raise ValueError(f"Unknown model service type: {kind}")