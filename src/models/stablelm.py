import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline


def something():
  # tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
  # model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
  # model.half().cuda()

  repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

  tokenizer = AutoTokenizer.from_pretrained(repo_id)
  model = AutoModelForCausalLM.from_pretrained(repo_id)
  model.cuda() # .half()


  class StopOnTokens(StoppingCriteria):
      def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
          stop_ids = [50278, 50279, 50277, 1, 0]
          for stop_id in stop_ids:
              if input_ids[0][-1] == stop_id:
                  return True
          return False

  system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
  - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
  - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
  - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
  - StableLM will refuse to participate in anything that could harm a human.
  """ 

  # prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"


  question = "Don't answer my question, just show 2 semantically similar messages to the following question: Who is planning to go on a beach vacation?"
  context = """
  Alice: Hey everyone, how was your weekend? Did anyone do anything fun?
  Bob: I went to the mountains for a hiking trip, it was amazing!
  Claire: I just stayed home and relaxed. I'm planning on going on a beach vacation soon though!
  David: Oh, that sounds like fun Claire. Where are you thinking of going?
  Alice: I went to Hawaii last year and it was incredible. You should definitely consider going there.
  Claire: That's actually where I'm planning on going! I heard the beaches there are amazing.
  Bob: I've never been to Hawaii, but I've been to the Caribbean a few times for beach vacations.
  David: I've actually never been on a beach vacation before. I usually prefer to travel to cities and explore the local culture.
  Alice: That's cool, David. Have you been to any interesting cities recently?
  Bob: Speaking of vacations, has anyone been following the news about the new theme park opening up in California?
  Claire: Oh, I heard about that! But let's stay on topic. So, Alice and I are interested in beach vacations. Bob, do you have any recommendations for beach destinations other than Hawaii?
  David: Yeah, I'd love to hear about some other options as well.
  Alice: I've heard great things about the beaches in Thailand and Bali. Those could be good options.
  Bob: I've been to both of those places, and they're definitely worth considering. But if you're looking for something closer to home, Florida has some amazing beaches too.
  Claire: Thanks for the suggestions, everyone! I'm definitely going to look into those places.
  David: Yeah, me too. I'm actually getting more excited about the idea of a beach vacation now.
  """

  prompt = f"{system_prompt}<|USER|>{question} {context}<|ASSISTANT|>"

  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  tokens = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.1,
    do_sample=True,
    stopping_criteria=StoppingCriteriaList([StopOnTokens()])
  )
  print(tokenizer.decode(tokens[0], skip_special_tokens=True))


  # question = "Don't answer my question, just show 2 semantically similar messages to the following question: Who is planning to go on a beach vacation?"
  # context = """
  # Alice: Hey everyone, how was your weekend? Did anyone do anything fun?
  # Bob: I went to the mountains for a hiking trip, it was amazing!
  # Claire: I just stayed home and relaxed. I'm planning on going on a beach vacation soon though!
  # David: Oh, that sounds like fun Claire. Where are you thinking of going?
  # Alice: I went to Hawaii last year and it was incredible. You should definitely consider going there.
  # Claire: That's actually where I'm planning on going! I heard the beaches there are amazing.
  # Bob: I've never been to Hawaii, but I've been to the Caribbean a few times for beach vacations.
  # David: I've actually never been on a beach vacation before. I usually prefer to travel to cities and explore the local culture.
  # Alice: That's cool, David. Have you been to any interesting cities recently?
  # Bob: Speaking of vacations, has anyone been following the news about the new theme park opening up in California?
  # Claire: Oh, I heard about that! But let's stay on topic. So, Alice and I are interested in beach vacations. Bob, do you have any recommendations for beach destinations other than Hawaii?
  # David: Yeah, I'd love to hear about some other options as well.
  # Alice: I've heard great things about the beaches in Thailand and Bali. Those could be good options.
  # Bob: I've been to both of those places, and they're definitely worth considering. But if you're looking for something closer to home, Florida has some amazing beaches too.
  # Claire: Thanks for the suggestions, everyone! I'm definitely going to look into those places.
  # David: Yeah, me too. I'm actually getting more excited about the idea of a beach vacation now.
  # """

  # prompt = f"{system_prompt}<|USER|>{question} {context}<|ASSISTANT|>"


# def get_model(to_langchain=True):
#     repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

#     tokenizer = AutoTokenizer.from_pretrained(repo_id)
#     model = AutoModelForCausalLM.from_pretrained(repo_id)
#     model.cuda() # .half()



from langchain import HuggingFacePipeline


def get_model(to_langchain=True):
    repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

    tokenizer = AutoTokenizer.from_pretrained(repo_id, max_new_tokens=100000, max_length=512)
    model = AutoModelForCausalLM.from_pretrained(repo_id)
    # model.cuda() # .half()

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, device='cuda:1'
    )
    llm = HuggingFacePipeline(pipeline=pipe)# max_new_tokens
    # llm = HuggingFacePipeline.from_model_id(model_id="thebloke/stable-vicuna-13B-HF", task="text-generation", model_kwargs={"temperature":0, "max_length":1024}, device=0)
    return llm


if __name__ == "__main__":
    model = get_model()

    question = "what is the meaning of fire in Greek philosophy?"

    answer = model(question)

    print(answer)
