# telegram loader
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from src.utils.telegram_chat_loader import TelegramChatLoader

from dotenv import load_dotenv

load_dotenv()


def get_model(repo_id=None) -> HuggingFacePipeline:
    model_id = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"
    # llm = HuggingFaceHub(repo_id=repo_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10
    )
    hf = HuggingFacePipeline(pipeline=pipe)
    return hf


def get_model_stablelm():
    repo_id = "stabilityai/stablelm-tuned-alpha-3b"
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})
    return llm


def get_documents(chat_file_path):
    loader = TelegramChatLoader(chat_file_path)
    documents = loader.load()
    return documents


# model = get_model_stablelm()
model = get_model()
documents = get_documents("/home/qazybek/NVME/repos/mlops/qa_llm/data/raw/chat_history.json")

chain  = load_qa_chain(llm=model, chain_type="map_reduce")
query = "что спросил Kazybek Askarbek?"

print(chain.run(input_documents=documents, question=query))
