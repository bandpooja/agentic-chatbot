import os
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import LLMChain
import pandas as pd
import json


OPENAI_KEY = os.environ["OPENAI_KEY"]

def load_excel():
    # excel_file = download_blob.download_from_container(connection_string, container_name)
    df = pd.read_excel("***")   # path to excel to create the FAISS DB for RAG with
    return df


def process_documents(df):
    df_records = df.drop(["RFP"], axis=1).to_dict('records')
    texts = [json.dumps(r) for r in df_records]
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    docs = [Document(page_content=t) for t in texts]
    return docs, texts


def create_faiss_index(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
    faiss_db = FAISS.from_texts(texts, embeddings)
    faiss_db.save_local("faiss_db")  # needs a faiss DB created locally for it to work
    print("saved db")


if __name__== "__main__":
    df = load_excel()
    _, texts = process_documents(df)
    db = create_faiss_index(texts)
