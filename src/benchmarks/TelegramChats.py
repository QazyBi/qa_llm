from src.utils import get_data_path


import random


class TelegramBenchmark:
    question_n_answers = {
        # extraction questions
        'what Kazybek Askarbek said?': 'Brat',

        # synthesis questions
        # 'how many people will go to camping?': 'three'
    }

    @staticmethod
    def get_context() -> str:
        with open(get_data_path() / 'raw' / 'chat_history.json', 'r') as f:
            return f.read()

    @staticmethod
    def get_random_sample():
        items = TelegramBenchmark.question_n_answers.items()
        return random.choice(list(items))