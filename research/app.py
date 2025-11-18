from pinecone import Pinecone
from dotenv import load_dotenv
import os 

load_dotenv()
pinecone_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_key)
index = pc.Index("medical-chatbot")


print(pc.list_indexes())