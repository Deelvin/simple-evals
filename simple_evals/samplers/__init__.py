from .chat_completion_sampler import ChatCompletionSampler

MODEL_REGISTRY = {
    "openai": ChatCompletionSampler,
}


def get_sampler(sampler_name):
    return MODEL_REGISTRY[sampler_name]
