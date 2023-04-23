import json

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


# 2. Load the chat history JSON file
with open("../data/raw/chat_history.json", "r") as f:
    chat_history = json.load(f)

# 3. Upload each message as a separate document to OpenAI

valid_entity = lambda e: e["text"] != "" and "from" in e

# get useful information from message metadata
documents = ({"text": entity["text"], "metadata": {"id": entity["id"], "date": entity["date"], "from": entity["from"]}}
             for entity in chat_history["messages"] if valid_entity(entity))

# get the model and the tokenizer
tokenizer = AutoTokenizer.from_pretrained("AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")
model = AutoModelForQuestionAnswering.from_pretrained("AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru")

# Example context and question
context = [str(doc) for doc in documents]
question = "что спрашивает Kazybek Askarbek?"
inputs = tokenizer(question, str(context[645:649]), return_tensors="pt")  # model can process only limited number of characters

outputs = model(**inputs)

# Get the start and end logits for the answer
start_logits, end_logits = outputs['start_logits'], outputs['end_logits']
start_logits = start_logits.squeeze(-1).tolist()
end_logits = end_logits.squeeze(-1).tolist()

# Decode the tokenized input to get the answer
answer_start = int(torch.argmax(torch.tensor(start_logits)))
answer_end = int(torch.argmax(torch.tensor(end_logits))) + 1
# !!! added +100 because model prints too few characters, fix later!!!
answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end+100]))

print(f"Expected: {context[647]}", end='\n\n\n')
print("Answer: ", answer)
