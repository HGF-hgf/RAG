import os
from dotenv import load_dotenv
from pymongo import MongoClient
from llama_index.llms.openai import OpenAI

load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGODB_URI)
db = client['cellphoneS']
phone_collection = db["phone"]
laptop_collection = db["laptop"]
tablet_collection = db["tablet"]

# LLM configuration
llm = OpenAI(model="gpt-4o-mini", temperature=0)