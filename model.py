from langchain_ollama import ChatOllama

def get_llm():
    return ChatOllama(model = "gemma4:e4b",                             #load and return Gemma 4 model from ollama
                      temperature = 0.6)