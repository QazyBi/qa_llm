from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader

from src.utils import get_data_path
# from src.question_answering.extractive_qa_huggingface import get_model
from src.question_answering.xlm_roberta_ru import get_model
# from src.question_answering.bloom import get_model
from src.embedders.instruct_embeddings import get_embedder
from langchain.vectorstores.faiss import FAISS


if __name__ == "__main__":
    loader = PyPDFLoader(str(get_data_path() / "raw" / "example.pdf"))
    documents = loader.load()[:10]
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embedder = get_embedder()
    # create the vectorestore to use as the index
    # db = Chroma.from_documents(texts, embeddings)
    db = FAISS.from_documents(documents, embedder)  # this can be read from pkl file

    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":3})

    model_type = "not  langchain" # not 
    query = "How many AI publications in 2021?"


    llm = get_model(False)

    if model_type == "langchain":
        """
        
            huggingfacepipeline supports only "text2text-generation", "text-generation" models, thus I need to pick powerful models for thats
        """
        # create a chain to answer questions 
        qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = qa({"query": query})
    else:

        rel_docs = retriever.get_relevant_documents(query)
        for doc in rel_docs:
            # print(doc.page_content)
            print(llm(question=query, context=doc.page_content))