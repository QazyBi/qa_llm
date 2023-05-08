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
# from src.models.stablelm import get_model
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
            with open("/home/qazybek/NVME/repos/mlops/qa_llm/vectorstore_django.pkl", "wb") as f:
                pickle.dump(vectorstore, f)

            return vectorstore
    else:
        raise NotImplementedError()
    

# %%

# def main():
# loader = PyPDFLoader(str(get_data_path() / "raw" / "example.pdf"))
# documents = loader.load()[:10]
vectorstore = get_vectorstore(load_cache=False)  # True

# expose this index in a retriever interface
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

# query = "How many AI publications in 2021?"
query, answer = TelegramBenchmark.get_random_sample()


# %%

to_langchain = True


# llm = get_model(to_langchain)

# ===
from transformers import AutoModelForCausalLM, LlamaTokenizer, pipeline
import torch
from langchain import HuggingFacePipeline

model_id = "thebloke/stable-vicuna-13B-HF"
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)  # , device_map="auto", load_in_8bit=False

def get_response(prompt, model):
    ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(ids, max_length=len(prompt) + 150)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

def format_prompt(question, context):
    return f"### Human: Given this text: \n\n{context}\n\nAnswer following question: {question}\n\n### Assistant:"

# your_prompt_here = "BEST SECTORS:  Healthcare (0.11%), Communication Svcs. (0.17%), Utilities (0.31%), Industrials (0.38%), Real Estate (0.59%) WORST SECTORS:  Energy (1.92%), Financials (1.19%), Materials (1.11%), Tech (0.83%), Consumer Spls. (0.79%), Consumer Disc. (0.71%) +11.6% GNRC (Generac):  Q1 EBITDA ahead and company maintained FY guidance; takeaways noted help from better C&I demand and margins; also some discussion about favorable HSB lead indicators and elevated Q1 outage activity and field inventory progress; expects Residential rebound in 2H. +7.0% LTHM (Livent Corp.):  Big Q1 EBITDA beat on strong pricing, while volumes saw a small decline; raised FY guidance; takeaways noted record levels of revenue and profitability, expectations for further price increased and a recovery in volume as year progresses, favorable contracts with 70% of prices fixed for 2023, strong lithium demand. +6.7% LLY (Eli Lilly):  Announced its phase 3 study of donenemab for Alzheimer's met primary and all secondary endpoints measuring cognitive/functional decline; said patients displayed ~35% slowing of decline while 47% of participants showed no decline; some concerns on safety profile vs BIIB's lecanemab. +4.7% CLX (Clorox):  Fiscal Q3 results strong and company raised FY EPS guidance by 6% at midpoint; takeaways focused on pricing power(price/mix +19%), margin expansion, more elastic than expected volumes (though still declined), improved execution; more cautious commentary revolved around valuation. +2.0% EMR (Emerson Electric):  Big fiscal Q2 EBITDA and EPS beat with revenues also ahead; organic growth accelerated to 14% from 6% in prior Q; order growth improved to 7% y/y from 5%; highlighted strong end market demand and excellent operational execution; raised FY guidance; noted no slowdown in orders despite improving lead times. -17.3% EL (Estée Lauder):  Fiscal Q3 EPS missed and company slashed FY guidance by more than 30% (third straight Q of lower FY23 guide); noted Asia travel retail business continued to be pressured by slower than anticipated recovery from Covid; flagged slower prestige beauty growth in China; inventory rebalancing in Hainan another issue. -12.3% SPR (Spirit AeroSystems Holdings):  Q1 loss larger than expected; takeaways focused on charges for the 737, 787, A350 and A220, driving another loss in the Commercial segment; company said it has identified the quality issue surrounding the 737 and has begun implementing repairs; expects work to be completed by end of July. -9.3% AMD (Advanced Micro Devices):  Q1 results better but company guided Q2 below; takeaways focused on weak PC and data center; company flagged continued inventory destocking for some bigger cloud customers and soft enterprise demand in the cloud; also some discussion about rising competition; however, expects strong data center ramp in 2H and reaffirmed PCs bottomed in Q1 and should start to improve; downgraded at BofA. -9.2% SBUX (Starbucks):  Fiscal Q2 results better with takeaways focused on China comp rebound and 12% increase in US comps; however, some disappointment company only reiterated FY guidance; takeaways noted caution surrounding more difficult macro backdrop, though company said no impact yet; also flagged recent normalization in US comp trends and more difficult near-term comparisons; however, lot of talk about how guidance looks conservative. -3.9% YUM (YUM! Brands):  Q1 earnings missed though revenue better; comp growth better across all brands; highlighted strong digital-sales trends; analysts noted some concerns about weaker margins (particularly at KFC)."

# # Test input text
# input_text = [f"### Human: Resume this text: \n\n{your_prompt_here}\n\n### Assistant:"][0]

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, device='cuda:1', max_new_tokens=10
)
llm = HuggingFacePipeline(pipeline=pipe)

# Tokenize the input text
# input_ids = Tokenizer.encode(input_text, return_tensors="pt")
# Generate output text
# print("Before generating text")
# output_ids = model.generate(input_ids, max_length=50)
# print("After generating text")

# Decode the output tokens
# output_text = Tokenizer.decode(output_ids[0], skip_special_tokens=True)

# ===
# %%

if to_langchain:
    """
    
        huggingfacepipeline supports only "text2text-generation", "text-generation" models, thus I need to pick powerful models for thats
    """
    # create a chain to answer questions 
    # %%
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # %%
    result = qa({"query": query})
    print(result)
    # %%
else:

    rel_docs = retriever.get_relevant_documents(query)
    print(f"Question: {query} - {answer}", end='\n\n')
    for doc in rel_docs:
        print(doc)
        # print(doc.page_content)
        print(llm(question=query, context=doc.page_content))
        print()


# if __name__ == "__main__":
#     main()
# %%
