from src.utils import get_data_path

from src.utils.telegram_chat_loader import TelegramChatLoader
import random


class TelegramBenchmark:
    question_n_answers = {
        # extraction questions
        # 'что сказал Kazybek Askarbek?': 'фю',
        # "какая IDE удобнее для работы с базами данных": "DataGrip"
        # "alpaca": "DataGrip",
        "проект с chatgpt": "Всем привет! запрыгиваю в последний вагон :)\nЕсли есть желающий присоедениться к проекту - вэлекам!\n\nПроект представляет собой создание AI-помощника, подобного ChatGPT, с дополнительным преимуществом - возможностью обучения его на описании бизнеса, процессов, с использованием вашей базы знаний. Он может быть использован в качестве помощника, отвечающего на вопросы по продукту, бизнесу или процессу. \n\nКроме того, его можно использовать как свою личную справочную - загрузить все свои билеты на самолеты, концерты, ссылки на интересные статьи и просто идеи, а затем запрашивать информацию, например, о датах поездки в Турцию или попросить показать все идеи стартапов, которые были записаны в этом году. \n\nПри ответе на вопрос помощник не только дает ответ, но и показывает \"доказательство\" - изначатьный документ или текст, который был добавлен в базу знаний пользователем.",
        # synthesis questions
        # 'how many people will go to camping?': 'three'
    }

    data_path = get_data_path() / 'raw' / 'chat_history.json'

    @staticmethod
    def get_context() -> str:
        with open(TelegramBenchmark.data_path, 'r') as f:
            return f.read()

    @staticmethod
    def get_random_sample():
        """Returns one question and its answer"""
        items = TelegramBenchmark.question_n_answers.items()
        return random.choice(list(items))
    
    @staticmethod
    def get_loader():
        return TelegramChatLoader(str(TelegramBenchmark.data_path))
    

    @staticmethod
    def get_documents():
        return TelegramBenchmark.get_loader().load()