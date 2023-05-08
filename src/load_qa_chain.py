"""Super slow and comsuming too much gpu, at least when bloom is used with AI report document

* better way is to use the RetrievalQA
"""

from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyPDFLoader

# from message_search.langchain import 
# from src.question_answering.extractive_qa_huggingface import get_model
from src.models.bloom import get_model
# from src.question_answering.xlm_roberta_ru import get_model
from src.utils.chat_data import get_documents
from src.utils import get_data_path


if __name__ == "__main__":
    # documents = get_documents("/home/qazybek/NVME/repos/mlops/qa_llm/data/raw/chat_history.json")
    # 
    loader = PyPDFLoader(str(get_data_path() / "raw" / "example.pdf"))
    documents = loader.load()[:1] 

    # print(len(documents))

    # print(documents)
    model = get_model()
    chain  = load_qa_chain(llm=model, chain_type="map_reduce")  # map_reduce

    query = "how many AI articles published?"
    
    from langchain import LLMChain, PromptTemplate

    prompt = PromptTemplate(
        input_variables=["question"],
    template="{question}")
    
    # llm_chain = LLMChain(prompt=prompt, llm=model)    
    # print(llm_chain.run(query))
    print(chain.run(input_documents=documents, question=query))
