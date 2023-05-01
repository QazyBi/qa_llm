from src.utils import get_data_path


import random


class CampingBenchmark:
    question_n_answers = {
        # extraction questions
        'who will bring cooler with ice?': 'Brat',

        # synthesis questions
        'how many people will go to camping?': 'three'
    }

    @staticmethod
    def get_context() -> str:
        with open(get_data_path() / 'raw' / 'camping_chat.txt', 'r') as f:
            return f.read()

    @staticmethod
    def get_random_sample():
        items = CampingBenchmark.question_n_answers.items()
        return random.choice(list(items))