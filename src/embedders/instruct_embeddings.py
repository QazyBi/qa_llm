"""
check for more: https://habr.com/ru/articles/669674/
llamacpp
cohere - paid

"""

from functools import partial

# pip install transformers sentencepiece
import torch
from transformers import AutoTokenizer, AutoModel, BertModel, BertTokenizerFast


def get_embedder():
    from langchain.embeddings import HuggingFaceInstructEmbeddings
    embeddings = HuggingFaceInstructEmbeddings(
            embed_instruction="Represent the each separate message for retrieval: ",
            query_instruction="represent the message"  # Represent the question for retrieving supporting texts from the messages: 
    )
    return embeddings


def get_embedder_hf():
    from langchain.embeddings import HuggingFaceEmbeddings

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cuda:0'}
    hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    return hf


def get_embedder_rubert():
    """https://huggingface.co/cointegrated/rubert-tiny2"""
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
    model.cuda()  # uncomment it if you have a GPU

    def embed_bert_cls(text, model, tokenizer):
        t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
                model_output = model(**{k: v.to(model.device) for k, v in t.items()})
        embeddings = model_output.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings)
        return embeddings[0].cpu().numpy()

    return partial(embed_bert_cls, model=model, tokenizer=tokenizer)


def get_embedder_labse():
    "https://huggingface.co/setu4993/LaBSE"
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE")
    model = model.eval()

    def process(sentence):
        english_inputs = tokenizer([sentence], return_tensors="pt", padding=True)

        with torch.no_grad():
            english_outputs = model(**english_inputs).pooler_output

        return english_outputs[0].numpy()  # 768
    
    return process


if __name__ == "__main__":
    # embedder = get_embedder()
    # embedder = get_embedder_rubert()
    embedder = get_embedder_labse()

#     text1 = "Энкодер предложений (sentence encoder) – это модель, которая сопоставляет коротким текстам векторы в многомерном пространстве, причём так, что у текстов, похожих по смыслу, и векторы тоже похожи. Обычно для этой цели используются нейросети, а полученные векторы называются эмбеддингами. Они полезны для кучи задач, например, few-shot классификации текстов, семантического поиска, или оценки качества перефразирования."
#     text2 = "Но некоторые из таких полезных моделей занимают очень много памяти или работают медленно, особенно на обычных CPU. Можно ли выбрать наилучший энкодер предложений с учётом качества, быстродействия, и памяти? Я сравнил 25 энкодеров на 8 задачах, и составил их рейтинг. Самой качественной моделью оказался mUSE, самой быстрой из предобученных – FastText, а по балансу скорости и качества победил rubert-tiny2. Код бенчмарка выложен в репозитории encodechka, а подробности – под катом."
#     text3 = "Немецкий фотограф Роберт Кнешке в феврале этого года узнал, что его фотографии использовались для обучения генераторов изображений. Благодаря ресурсу Have I Been Trained? Кнешке вышел на LAION-5B, датасет из более чем 5,8 млрд изображений, принадлежащий известной некоммерческой организации LAION. Именно её данные использовала Stability AI для обучения Stable Diffusion. Кнешке обнаружил «кучу изображений» из его портфолио в LAION-5B и написал об этом в своём блоге."
    
    text5 = "описание проекта с chatgpt"
    text4 = "проект с chatgpt"
    text1 = "Всем привет! запрыгиваю в последний вагон :)\nЕсли есть желающий присоедениться к проекту - вэлекам!\n\nПроект представляет собой создание AI-помощника, подобного ChatGPT, с дополнительным преимуществом - возможностью обучения его на описании бизнеса, процессов, с использованием вашей базы знаний. Он может быть использован в качестве помощника, отвечающего на вопросы по продукту, бизнесу или процессу. \n\nКроме того, его можно использовать как свою личную справочную - загрузить все свои билеты на самолеты, концерты, ссылки на интересные статьи и просто идеи, а затем запрашивать информацию, например, о датах поездки в Турцию или попросить показать все идеи стартапов, которые были записаны в этом году. \n\nПри ответе на вопрос помощник не только дает ответ, но и показывает \"доказательство\" - изначатьный документ или текст, который был добавлен в базу знаний пользователем."        
    text2 = "Самый простой и самый известный - уже подключены к ChatGpt ?  В него вставляешь - \"прошу реализовать парсинг сайта "
    text3 = "Причём, как я понял эт же очень просто. Пишешь ChatGPT, напиши код для получения обучающего датасета, обучения модели и инференса, чтобы получился чат жпт и запускаешь. Все. Датасаентисты не нужны больше нафиг 😂"

    texts = [text1, text2, text3, text4, text5]

    from itertools import permutations, combinations
    from sklearn.metrics.pairwise import cosine_similarity
    
    # embeddings = [embedder.embed_query(text.lower()) for text in texts]
    # embeddings = [embedder(text.lower()) for text in texts]
    perms = combinations(texts, 2)

    for t1, t2 in perms:
        print("====")
        print(f"Text1: {t1} \nText2: {t2}")
        # print(cosine_similarity([embedder.embed_query(t1)], [embedder.embed_query(t2)]))
        print(cosine_similarity([embedder(t1)], [embedder(t2)]))
        print("=====", end="\n\n")