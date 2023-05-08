
# todo:
"""

- llama

https://python.langchain.com/en/latest/modules/chains/index_examples/vector_db_qa.html

thing I need will need soon(evaluation of different settings of models):
    - https://github.com/rlancemartin/auto-evaluator


fine intro:
    - https://medium.com/nlplanet/two-minutes-nlp-quick-intro-to-question-answering-124a0930577c
    - need to follow links to understand different approaches
"""


# load a model
# pass it with sample text
# evaluate the result
import torch
# from src.models.bloom import get_model as get_model_bloom
from src.models.extractive_qa_huggingface import get_model as get_model_extractive
from src.models.xlm_roberta_ru import get_model as get_model_xml


options = {
    'extractive_qa': get_model_extractive,
    # 'xlm-roberta-ru': get_model_xml
}


if __name__ == "__main__":
    device = torch.device("cuda:0")

    # prompt = """Даны тексты, скажи какие из них наиболее похожие друг другу?
    # text1 = "Всем привет! запрыгиваю в последний вагон :)\nЕсли есть желающий присоедениться к проекту - вэлекам!\n\nПроект представляет собой создание AI-помощника, подобного ChatGPT, с дополнительным преимуществом - возможностью обучения его на описании бизнеса, процессов, с использованием вашей базы знаний. Он может быть использован в качестве помощника, отвечающего на вопросы по продукту, бизнесу или процессу. \n\nКроме того, его можно использовать как свою личную справочную - загрузить все свои билеты на самолеты, концерты, ссылки на интересные статьи и просто идеи, а затем запрашивать информацию, например, о датах поездки в Турцию или попросить показать все идеи стартапов, которые были записаны в этом году. \n\nПри ответе на вопрос помощник не только дает ответ, но и показывает \"доказательство\" - изначатьный документ или текст, который был добавлен в базу знаний пользователем."        
    # text2 = "Самый простой и самый известный - уже подключены к ChatGpt ?  В него вставляешь - \"прошу реализовать парсинг сайта "
    # text3 = "Причём, как я понял эт же очень просто. Пишешь ChatGPT, напиши код для получения обучающего датасета, обучения модели и инференса, чтобы получился чат жпт и запускаешь. Все. Датасаентисты не нужны больше нафиг 😂"
    # text4 = "проект с chatgpt"
    # """


#     prompt = "Какой из текстов похож этому: проект с chatgpt"
#     context = """
# text1 = "Всем привет! запрыгиваю в последний вагон :)\nЕсли есть желающий присоедениться к проекту - вэлекам!\n\nПроект представляет собой создание AI-помощника, подобного ChatGPT, с дополнительным преимуществом - возможностью обучения его на описании бизнеса, процессов, с использованием вашей базы знаний. Он может быть использован в качестве помощника, отвечающего на вопросы по продукту, бизнесу или процессу. \n\nКроме того, его можно использовать как свою личную справочную - загрузить все свои билеты на самолеты, концерты, ссылки на интересные статьи и просто идеи, а затем запрашивать информацию, например, о датах поездки в Турцию или попросить показать все идеи стартапов, которые были записаны в этом году. \n\nПри ответе на вопрос помощник не только дает ответ, но и показывает \"доказательство\" - изначатьный документ или текст, который был добавлен в базу знаний пользователем."        
# text2 = "Самый простой и самый известный - уже подключены к ChatGpt ?  В него вставляешь - \"прошу реализовать парсинг сайта "
# text3 = "Причём, как я понял эт же очень просто. Пишешь ChatGPT, напиши код для получения обучающего датасета, обучения модели и инференса, чтобы получился чат жпт и запускаешь. Все. Датасаентисты не нужны больше нафиг 😂"
# """

#     prompt_text = "Tell me about your favorite vacation spot."
#     context = """
# text A: "Although I don't have a favorite vacation spot, I do enjoy exploring new places and experiencing different cultures. Last summer, I went on a hiking trip to the Grand Canyon, which was a challenging but rewarding adventure."
# text B: "I just started learning how to cook Indian cuisine and I'm really enjoying the process. The combination of different spices and flavors is fascinating to me."
# text C: "My go-to vacation spot is a small beach town in Mexico. The white sand, crystal clear waters, and delicious seafood always make for a perfect getaway."
# """

#     prompt_text = "What's your favorite way to relax after a long day?"

#     context = """
# text A: "I recently started taking salsa dance lessons and it's become my favorite hobby. The energy and passion of the dance is exhilarating and helps me forget about any worries."
# text B: "After a long day, I love to unwind by taking a hot bath and reading a book. It's a peaceful way to let go of the stresses of the day."
# text C: "I don't necessarily have a favorite way to relax after a long day, but I do enjoy catching up with friends over a drink or watching a movie with my family."
# """

    prompt_text = "Who is planning to go on a beach vacation?"

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

    # prompt = f"Answer the following question: {prompt_text}"
    # prompt = f"Show relevant messages to the following question: {prompt_text}"
    # prompt = f"Can you find me the message where someone talks about {prompt_text}"
    

#     context = """
# Элис: Привет всем, как прошли выходные? Кто-нибудь занимался чем-то интересным?
# Боб: Я отправился в горы на поход, было потрясающе!
# Клэр: Я просто оставалась дома и отдыхала. Но я планирую отправиться на пляжный отдых скоро!
# Дэвид: О, это звучит здорово, Клэр. Куда ты думаешь отправиться?
# Элис: Я была на Гавайях в прошлом году, и это было невероятно. Тебе определенно стоит туда поехать.
# Клэр: Вот туда я и планирую отправиться! Я слышала, что там пляжи просто потрясающие.
# Боб: Я никогда не был на Гавайях, но я несколько раз отправлялся на пляжный отдых в Карибский регион.
# Дэвид: На самом деле я никогда не отправлялся на пляжный отдых. Я обычно предпочитаю путешествовать в города и изучать местную культуру.
# Элис: Это круто, Дэвид. Ты был в каких-нибудь интересных городах в последнее время?
# Боб: Говоря о отпусках, кто-нибудь следит за новостями о новом тематическом парке, который открывается в Калифорнии?
# Клэр: О, я слышала об этом! Но давайте останемся в теме. Так вот, Элис и я интересуемся пляжным отдыхом. Боб, у тебя есть рекомендации для пляжных направлений, кроме Гавайев?
# Дэвид: Да, мне было бы интересно услышать о других вариантах тоже.
# Элис: Я слышала много хорошего о пляжах в Таиланде и на Бали. Возможно, это будет хорошим выбором.
# Боб: Я был и там, и там, и это определенно стоит рассмотрения. Но если вы ищете что-то ближе к дому, то во Флориде тоже есть потрясающие пляжи.
# Клэр: Спасибо за рекомендации, ребята! Я точно буду рассматривать эти места    
#     """

#     prompt = "Какие сообщения наиоболее похожи на следующий вопрос: Кто планирует отдыхать на пляже?"

    prompt = f"Which messages are the most semantically similar following question: {prompt_text}"

    for name, model_getter in options.items():
        model = model_getter(False)
        print(prompt)

        pred_answer = model(question=prompt, context=context)

        print(pred_answer)
