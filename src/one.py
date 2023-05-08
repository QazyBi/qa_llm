# %%
import sys

sys.path.append('/home/qazybek/NVME/repos/mlops/qa_llm')
import pickle
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from pathlib import Path
from langchain.vectorstores.faiss import FAISS

from src.utils import get_data_path
# from src.models.extractive_qa_huggingface import get_model
from src.models.stablelm import get_model
# from src.question_answering.xlm_roberta_ru import get_model
# from src.question_answering.bloom import get_model
from src.embedders.instruct_embeddings import get_embedder
from src.data.TelegramChats import TelegramBenchmark


def get_vectorstore(type_="telegram", load_cache=True):
    """
        TODO:
            [ ] try different embedders
            [ ] try different vectorstores


            [ ] try different text splitters 
                # split the documents into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.split_documents(documents)
    """
    if type_ == "telegram":
        if load_cache and Path("/home/qazybek/NVME/repos/mlops/qa_llm/vectorstore.pkl").exists():
            with open("/home/qazybek/NVME/repos/mlops/qa_llm/vectorstore.pkl", "rb") as f:
                vectorstore = pickle.load(f)
                return vectorstore
        else:
            documents = TelegramBenchmark.get_documents()

            embedder = get_embedder()
            # create the vectorestore to use as the index
            # db = Chroma.from_documents(texts, embeddings)
            vectorstore = FAISS.from_documents(documents, embedder)  # this can be read from pkl file

            # Save vectorstore
            with open("/home/qazybek/NVME/repos/mlops/qa_llm/vectorstore.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            return vectorstore
    else:
        raise NotImplementedError()

# %%
loader = PyPDFLoader(str(get_data_path() / "raw" / "example.pdf"))
documents = loader.load()[:10]

# %%
text_splitter = CharacterTextSplitter(separator='\n', chunk_size=256, chunk_overlap=0)
texts = text_splitter.split_documents(documents)


# %%
embedder = get_embedder()

# %%
vectorstore = FAISS.from_documents(texts, embedder)

# %%
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})

# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline

repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

tokenizer = AutoTokenizer.from_pretrained(repo_id, max_new_tokens=100000, max_length=512)
model = AutoModelForCausalLM.from_pretrained(repo_id)
# model.cuda() # .half()

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device='cuda:0', max_new_tokens=1000, temperature=0.1
)

from langchain import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=pipe)

# %%
qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=True)
# %%
query = "how many AI publications?"
result = qa({"query": query})

from pprint import pprint

pprint(result)
# %%
