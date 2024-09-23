from factscore.clm import CLM
from factscore.openai_lm import OpenAIModel
from factscore.google_lm import GoogleGeminiModel

class ModelFactory:
    LLAMA_MODELS = {
        "llama-2-7b": CLM,
        "llama-2-7b-chat": CLM,
        "llama-2-13b": CLM,
        "llama-2-13b-chat": CLM,
        "llama-2-70b": CLM,
        "llama-2-70b-chat": CLM,
        "llama-2-7b-hf": CLM,
        "llama-2-7b-chat-hf": CLM,
        "llama-2-13b-hf": CLM,
        "llama-2-13b-chat-hf": CLM,
        "llama-2-70b-hf": CLM,
        "llama-2-70b-chat-hf": CLM,
        "meta-llama-3-8b": CLM,
        "meta-llama-3-8b-instruct": CLM,
        "meta-llama-3-70b": CLM,
        "meta-llama-3-70b-instruct": CLM,
        "meta-llama-3.1-8b": CLM,
        "meta-llama-3.1-8b-instruct": CLM,
        "meta-llama-3.1-70b": CLM,
        "meta-llama-3.1-70b-instruct": CLM
    }

    OpenAI_MODELS = {
        "gpt-3.5-turbo": OpenAIModel,
        "gpt-3.5-turbo-0125": OpenAIModel,
        "gpt-4-0125-preview": OpenAIModel,
        "gpt-4-turbo-preview": OpenAIModel,
        "gpt-4o": OpenAIModel
    }

    GEMINI_MODELS = {
        "gemini-1.0-pro": GoogleGeminiModel,
        "gemini-1.5-pro": GoogleGeminiModel
    }

    OTHER_MODELS = {
        "exaone-3.0-7.8b-instruct": CLM
    }

    MODEL_MAPPING = {**LLAMA_MODELS, **OpenAI_MODELS, **GEMINI_MODELS, **OTHER_MODELS}

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        model_name = model_name_or_path.split('/')[-1].lower()

        if model_name in cls.MODEL_MAPPING:
            model_class = cls.MODEL_MAPPING[model_name]
            print(f'Model loading... {model_name}')
            return model_class.from_pretrained(model_name_or_path, **kwargs)
        else:
            raise ValueError(f"Model {model_name} not found in MODEL_MAPPING.")
