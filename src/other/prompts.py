from langchain.prompts import PromptTemplate


# todo:
#   prompt templates - useful to align the LLM behaviour
#   prompt examples + selectors - useful to align the LLM behaviour

question_prompt = PromptTemplate(
    input_variables=["question"],
    template="Based on chat history answer following question: {question}?",
)

print(question_prompt.format(question="who owns this chat"))
