from langchain import HuggingFacePipeline


def get_model():
    llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", task="text-generation", model_kwargs={"temperature":0, "max_length":512}, device=0)
    return llm


if __name__ == "__main__":
    model = get_model()

    question = "what is the meaning of fire in Greek philosophy?"

    answer = model(question)

    print(answer)
