from langchain.embeddings import HuggingFaceInstructEmbeddings


def get_embedder():
    embeddings = HuggingFaceInstructEmbeddings(
            embed_instruction="Represent the each separate message for retrieval: ",
            query_instruction="Represent the question for retrieving supporting texts from the messages: "
    )
    return embeddings