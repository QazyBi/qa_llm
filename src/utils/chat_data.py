from .telegram_chat_loader import TelegramChatLoader



def get_documents(chat_file_path):
    loader = TelegramChatLoader(chat_file_path)
    documents = loader.load()
    return documents
