import torch
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from src.data.CampingBenchmark import CampingBenchmark

from src.utils import get_logger

logger = get_logger()


def get_model(to_langchain=True) -> HuggingFacePipeline:
    logger.info("using huggingface standard qa")
    DEVICE = torch.device("cuda:0")
    qa_model_pipe = pipeline("question-answering", device=DEVICE, return_full_text=True)
    if to_langchain:
        return HuggingFacePipeline(pipeline=qa_model_pipe)
    else:
        return qa_model_pipe
        # raise NotImplementedError()


if __name__ == "__main__":
    model = get_model(False)

    question, answer = CampingBenchmark.get_random_sample()
    context = CampingBenchmark.get_context()

    output = model(question = question, context  = context)
    print(question, answer)
    print(output)
