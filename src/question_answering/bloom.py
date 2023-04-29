from langchain import HuggingFacePipeline


def get_model():
    llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", task="text-generation", model_kwargs={"temperature":0, "max_length":20000}, device=0)
    return llm