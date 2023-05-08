# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline
# from langchain import HuggingFacePipeline


# # def something():
# #   # tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
# #   # model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-7b")
# #   # model.half().cuda()

# #   repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

# #   tokenizer = AutoTokenizer.from_pretrained(repo_id)
# #   model = AutoModelForCausalLM.from_pretrained(repo_id)
# #   model.cuda() # .half()


# #   class StopOnTokens(StoppingCriteria):
# #       def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
# #           stop_ids = [50278, 50279, 50277, 1, 0]
# #           for stop_id in stop_ids:
# #               if input_ids[0][-1] == stop_id:
# #                   return True
# #           return False

# #   system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
# #   - StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
# #   - StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
# #   - StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
# #   - StableLM will refuse to participate in anything that could harm a human.
# #   """ 

# #   # prompt = f"{system_prompt}<|USER|>What's your mood today?<|ASSISTANT|>"


# #   question = "Don't answer my question, just show 2 semantically similar messages to the following question: Who is planning to go on a beach vacation?"
# #   context = """
# #   Alice: Hey everyone, how was your weekend? Did anyone do anything fun?
# #   Bob: I went to the mountains for a hiking trip, it was amazing!
# #   Claire: I just stayed home and relaxed. I'm planning on going on a beach vacation soon though!
# #   David: Oh, that sounds like fun Claire. Where are you thinking of going?
# #   Alice: I went to Hawaii last year and it was incredible. You should definitely consider going there.
# #   Claire: That's actually where I'm planning on going! I heard the beaches there are amazing.
# #   Bob: I've never been to Hawaii, but I've been to the Caribbean a few times for beach vacations.
# #   David: I've actually never been on a beach vacation before. I usually prefer to travel to cities and explore the local culture.
# #   Alice: That's cool, David. Have you been to any interesting cities recently?
# #   Bob: Speaking of vacations, has anyone been following the news about the new theme park opening up in California?
# #   Claire: Oh, I heard about that! But let's stay on topic. So, Alice and I are interested in beach vacations. Bob, do you have any recommendations for beach destinations other than Hawaii?
# #   David: Yeah, I'd love to hear about some other options as well.
# #   Alice: I've heard great things about the beaches in Thailand and Bali. Those could be good options.
# #   Bob: I've been to both of those places, and they're definitely worth considering. But if you're looking for something closer to home, Florida has some amazing beaches too.
# #   Claire: Thanks for the suggestions, everyone! I'm definitely going to look into those places.
# #   David: Yeah, me too. I'm actually getting more excited about the idea of a beach vacation now.
# #   """

# #   prompt = f"{system_prompt}<|USER|>{question} {context}<|ASSISTANT|>"

# #   inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
# #   tokens = model.generate(
# #     **inputs,
# #     max_new_tokens=64,
# #     temperature=0.1,
# #     do_sample=True,
# #     stopping_criteria=StoppingCriteriaList([StopOnTokens()])
# #   )
# #   print(tokenizer.decode(tokens[0], skip_special_tokens=True))


# #   # question = "Don't answer my question, just show 2 semantically similar messages to the following question: Who is planning to go on a beach vacation?"
# #   # context = """
# #   # Alice: Hey everyone, how was your weekend? Did anyone do anything fun?
# #   # Bob: I went to the mountains for a hiking trip, it was amazing!
# #   # Claire: I just stayed home and relaxed. I'm planning on going on a beach vacation soon though!
# #   # David: Oh, that sounds like fun Claire. Where are you thinking of going?
# #   # Alice: I went to Hawaii last year and it was incredible. You should definitely consider going there.
# #   # Claire: That's actually where I'm planning on going! I heard the beaches there are amazing.
# #   # Bob: I've never been to Hawaii, but I've been to the Caribbean a few times for beach vacations.
# #   # David: I've actually never been on a beach vacation before. I usually prefer to travel to cities and explore the local culture.
# #   # Alice: That's cool, David. Have you been to any interesting cities recently?
# #   # Bob: Speaking of vacations, has anyone been following the news about the new theme park opening up in California?
# #   # Claire: Oh, I heard about that! But let's stay on topic. So, Alice and I are interested in beach vacations. Bob, do you have any recommendations for beach destinations other than Hawaii?
# #   # David: Yeah, I'd love to hear about some other options as well.
# #   # Alice: I've heard great things about the beaches in Thailand and Bali. Those could be good options.
# #   # Bob: I've been to both of those places, and they're definitely worth considering. But if you're looking for something closer to home, Florida has some amazing beaches too.
# #   # Claire: Thanks for the suggestions, everyone! I'm definitely going to look into those places.
# #   # David: Yeah, me too. I'm actually getting more excited about the idea of a beach vacation now.
# #   # """

# #   # prompt = f"{system_prompt}<|USER|>{question} {context}<|ASSISTANT|>"


# # # def get_model(to_langchain=True):
# # #     repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

# # #     tokenizer = AutoTokenizer.from_pretrained(repo_id)
# # #     model = AutoModelForCausalLM.from_pretrained(repo_id)
# # #     model.cuda() # .half()


# def get_model(to_langchain=True):
#     repo_id = "thebloke/stable-vicuna-13B-HF"  # "carperai/stable-vicuna-13b-delta"

#     tokenizer = AutoTokenizer.from_pretrained(repo_id, max_new_tokens=100000, max_length=512)
#     model = AutoModelForCausalLM.from_pretrained(repo_id)
#     # model.cuda() # .half()

#     pipe = pipeline(
#         "text-generation", model=model, tokenizer=tokenizer, device='cuda:1'
#     )
#     llm = HuggingFacePipeline(pipeline=pipe)# max_new_tokens
#     # llm = HuggingFacePipeline.from_model_id(model_id="thebloke/stable-vicuna-13B-HF", task="text-generation", model_kwargs={"temperature":0, "max_length":1024}, device=0)
#     return llm


# if __name__ == "__main__":
#     model = get_model()

# #     question = "what is the meaning of fire in Greek philosophy?"

# #     answer = model(question)

# #     print(answer)


from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch


model_id = "thebloke/stable-vicuna-13B-HF"
Tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)  # , device_map="auto", load_in_8bit=False)


def get_response(prompt, model):
    ids = Tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(ids, max_length=len(prompt) + 150)
    output_text = Tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


def format_prompt(question, context):
    return f"### Human: Given this text: \n\n{context}\n\nAnswer following question: {question}\n\n### Assistant:"

your_prompt_here = "BEST SECTORS:  Healthcare (0.11%), Communication Svcs. (0.17%), Utilities (0.31%), Industrials (0.38%), Real Estate (0.59%) WORST SECTORS:  Energy (1.92%), Financials (1.19%), Materials (1.11%), Tech (0.83%), Consumer Spls. (0.79%), Consumer Disc. (0.71%) +11.6% GNRC (Generac):  Q1 EBITDA ahead and company maintained FY guidance; takeaways noted help from better C&I demand and margins; also some discussion about favorable HSB lead indicators and elevated Q1 outage activity and field inventory progress; expects Residential rebound in 2H. +7.0% LTHM (Livent Corp.):  Big Q1 EBITDA beat on strong pricing, while volumes saw a small decline; raised FY guidance; takeaways noted record levels of revenue and profitability, expectations for further price increased and a recovery in volume as year progresses, favorable contracts with 70% of prices fixed for 2023, strong lithium demand. +6.7% LLY (Eli Lilly):  Announced its phase 3 study of donenemab for Alzheimer's met primary and all secondary endpoints measuring cognitive/functional decline; said patients displayed ~35% slowing of decline while 47% of participants showed no decline; some concerns on safety profile vs BIIB's lecanemab. +4.7% CLX (Clorox):  Fiscal Q3 results strong and company raised FY EPS guidance by 6% at midpoint; takeaways focused on pricing power(price/mix +19%), margin expansion, more elastic than expected volumes (though still declined), improved execution; more cautious commentary revolved around valuation. +2.0% EMR (Emerson Electric):  Big fiscal Q2 EBITDA and EPS beat with revenues also ahead; organic growth accelerated to 14% from 6% in prior Q; order growth improved to 7% y/y from 5%; highlighted strong end market demand and excellent operational execution; raised FY guidance; noted no slowdown in orders despite improving lead times. -17.3% EL (Estée Lauder):  Fiscal Q3 EPS missed and company slashed FY guidance by more than 30% (third straight Q of lower FY23 guide); noted Asia travel retail business continued to be pressured by slower than anticipated recovery from Covid; flagged slower prestige beauty growth in China; inventory rebalancing in Hainan another issue. -12.3% SPR (Spirit AeroSystems Holdings):  Q1 loss larger than expected; takeaways focused on charges for the 737, 787, A350 and A220, driving another loss in the Commercial segment; company said it has identified the quality issue surrounding the 737 and has begun implementing repairs; expects work to be completed by end of July. -9.3% AMD (Advanced Micro Devices):  Q1 results better but company guided Q2 below; takeaways focused on weak PC and data center; company flagged continued inventory destocking for some bigger cloud customers and soft enterprise demand in the cloud; also some discussion about rising competition; however, expects strong data center ramp in 2H and reaffirmed PCs bottomed in Q1 and should start to improve; downgraded at BofA. -9.2% SBUX (Starbucks):  Fiscal Q2 results better with takeaways focused on China comp rebound and 12% increase in US comps; however, some disappointment company only reiterated FY guidance; takeaways noted caution surrounding more difficult macro backdrop, though company said no impact yet; also flagged recent normalization in US comp trends and more difficult near-term comparisons; however, lot of talk about how guidance looks conservative. -3.9% YUM (YUM! Brands):  Q1 earnings missed though revenue better; comp growth better across all brands; highlighted strong digital-sales trends; analysts noted some concerns about weaker margins (particularly at KFC)."

# Test input text
input_text = [f"### Human: Resume this text: \n\n{your_prompt_here}\n\n### Assistant:"][0]

# Tokenize the input text
input_ids = Tokenizer.encode(input_text, return_tensors="pt")
# Generate output text
# print("Before generating text")
# output_ids = model.generate(input_ids, max_length=50)
# print("After generating text")

# Decode the output tokens
# output_text = Tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print("Generated text:", output_text)
# # Check if GPU is available and set the device accordingly
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
