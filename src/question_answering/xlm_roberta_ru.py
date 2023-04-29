from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

from src.benchmarks.CampingBenchmark import CampingBenchmark

def get_model(to_langchain=False):
    model_id = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    pipe = pipeline(
        "question-answering", #model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    if to_langchain:
        hf = HuggingFacePipeline(pipeline=pipe)
        return hf
    else:
        return pipe

# def get_model():
#     # tokenizer = AutoTokenizer.from_pretrained(model_id)
#     # model = AutoModelForQuestionAnswering.from_pretrained(model_id)
#     # pipe = pipeline(
#     #     "question-answering", #model=model, tokenizer=tokenizer, max_new_tokens=10
#     # )
#     # hf = HuggingFacePipeline(pipeline=pipe)
#     llm = HuggingFacePipeline.from_model_id(model_id="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru", task="question-answering", device=0)
#     return llm


if __name__ == "__main__":
    model = get_model()
    question, answer = CampingBenchmark.get_random_sample()
    context = CampingBenchmark.get_context()

    output = model(question = question, context  = context)
    print(question, answer)
    print(output)
