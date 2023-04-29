# https://github.com/vidalmaxime/chat-langchain-telegram/tree/main
from pathlib import Path

import pickle
# from omegaconf import DictConfig
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
# from langchain.document_loaders import TelegramChatLoader
from telegram_chat_loader import TelegramChatLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter


# cfg = DictConfig({
#     "text_splitter": "character",
# })
class Cfg:
    text_splitter = "character"

cfg = Cfg()

def embed(chat_file_path: Path = Path('data/raw/chat_history.json')):
    # get text_splitter
    if cfg.text_splitter == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size = 100,
            chunk_overlap  = 20,
            length_function = len,
        )
    elif cfg.text_splitter == "character":
        text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=512, chunk_overlap=20)
    else:
        raise ValueError()

    # get loader
    loader = TelegramChatLoader(chat_file_path)
    documents = loader.load()
    # documents = text_splitter.split_documents(loader.load())

    # Load Data to vectorstore
    # embeddings = OpenAIEmbeddings()
    from langchain.embeddings import HuggingFaceInstructEmbeddings

    embeddings = HuggingFaceInstructEmbeddings(
            embed_instruction="Represent the each separate message for retrieval: ",
            query_instruction="Represent the question for retrieving supporting texts from the messages: "
    )
    vectorstore = FAISS.from_documents(documents, embeddings)


    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


# from query_data import get_chain

# https://huggingface.co/datasets/calmgoose/book-embeddings
# https://github.com/HKUNLP/instructor-embedding#Installation


# thing to try: https://github.com/jerryjliu/llama_index
# thing to try: https://minigpt-4.github.io/


if __name__ == "__main__":
    # embed(Path('/home/qazybek/NVME/repos/mlops/qa_llm/result.json'))
    # embed(Path('/home/qazybek/NVME/repos/mlops/qa_llm/data/raw/chat_history.json'))
    # similarity search
    with open("vectorstore.pkl", "rb") as f:
        docsearch = pickle.load(f)

    question = "Данные(спутниковые снимки) кидают с s3, мне надо сначала скачать их, сделать препроцессинг, сделать инференс, сделать постпроцессинг и потом в виде архива загрузить куда-то. Мог бы поднять какой-то REST API сервис для именно этой модели, но проблема в том, что если таких моделей много то у меня будет большой список сервисов. "
    search = docsearch.similarity_search(question, k=3)

    for item in search:
        print(item.page_content)
        # print(f"From page: {item.metadata['page']}")
        print("---")

# #     qa_chain = get_chain(vectorstore)
# #     chat_history = []
# #     print("Chat with your docs!")
# #     while True:
# #         print("Human:")
# #         question = input()
# #         result = qa_chain({"question": question, "chat_history": chat_history})
# #         chat_history.append((question, result["answer"]))
# #         print("AI:")
# #         print(result["answer"])



# if cfg.text_splitter == "recursive":
#     text_splitter = RecursiveCharacterTextSplitter(
#         # Set a really small chunk size, just to show.
#         chunk_size = 100,
#         chunk_overlap  = 20,
#         length_function = len,
#     )
# elif cfg.text_splitter == "character":
#     text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=512, chunk_overlap=20)
# else:
#     raise ValueError()

# # get loader
# loader = TelegramChatLoader(Path('/home/qazybek/NVME/repos/mlops/qa_llm/result.json'))
# documents = text_splitter.split_documents(loader.load())
