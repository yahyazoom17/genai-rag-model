from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
import os
from dotenv import load_dotenv

app = FastAPI()

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,                # List of allowed origins
    allow_credentials=True,               # Allow cookies or authentication headers
    allow_methods=["*"],                  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],                  # Allow all headers
)

class ChatModel(BaseModel):
    input: str

embeddings = download_hugging_face_embeddings()

index_name = "testbot"

# Embed each chunk and upsert the embeddings into your Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
index_name=index_name,
embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4, max_tokens=500)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.get("/")
def get_home():
    return {"message":"RAG Model GenAI API"}

@app.post("/question")
def ask_model(question:ChatModel):
    data = question.dict()
    print(data)
    response = rag_chain.invoke(data)
    print(f"ChatBot: {response["answer"]}")
    return response