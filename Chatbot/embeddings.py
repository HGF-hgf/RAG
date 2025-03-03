import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY

def get_embeddings(text, model="text-embedding-ada-002"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding