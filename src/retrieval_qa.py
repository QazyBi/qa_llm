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

def main():
    # loader = PyPDFLoader(str(get_data_path() / "raw" / "example.pdf"))
    # documents = loader.load()[:10]
    vectorstore = get_vectorstore(load_cache=True)

    # expose this index in a retriever interface
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

    # query = "How many AI publications in 2021?"
    query, answer = TelegramBenchmark.get_random_sample()

    to_langchain = True

    
    llm = get_model(to_langchain)

    if to_langchain:
        """
        
            huggingfacepipeline supports only "text2text-generation", "text-generation" models, thus I need to pick powerful models for thats
        """
        # create a chain to answer questions 
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = qa({"query": query})
        print(result)
    else:

        rel_docs = retriever.get_relevant_documents(query)
        print(f"Question: {query} - {answer}", end='\n\n')
        for doc in rel_docs:
            print(doc)
            # print(doc.page_content)
            print(llm(question=query, context=doc.page_content))
            print()


if __name__ == "__main__":
    main()