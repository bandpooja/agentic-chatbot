from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.embeddings.openai import OpenAIEmbeddings
# from azure.storage.blob import ContainerClient
import requests
import time
import json
from PIL import Image
import os
import uuid
import streamlit as st
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.vectorstores import FAISS

python_tool = PythonREPLTool(
    description= 'A Python shell. Use this to execute python code. Also, use this to perform some analysis on data like create graphs.'
)
global chat_history_for_chain
chat_history_for_chain = ChatMessageHistory()
# leonardo image generation keys
LEO_IMG_TOKEN = os.environ["LEO_IMG_TOKEN"]
LEO_MODEL_ID = os.environ["LEO_MODEL_ID"]
LEO_STYLE_UUID = os.environ["LEO_STYLE_UUID"]
# openAI key
OPENAI_KEY = os.environ["OPENAI_KEY"]
# AWS Bedrock credentials
aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
aws_access_key_secret = os.environ["AWS_ACCESS_KEY_SECRET"]
aws_region = os.environ["AWS_REGION"]
aws_model_id = os.environ["AWS_MODEL_ID"]
# Azure OpenAI secrets
azure_deployment = os.environ["AZURE_DEPLOYMENT"]  # or your deployment
azure_api_version = os.environ["AZURE_MODEL_VERSION"] # or your api version
azure_openai_key = os.environ["AZURE_OPENAI_KEY"]
azure_endpoint = os.environ["AZURE_ENDPOINT"]


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# from azure.storage.blob import ContainerClient
# CONNECTION_STRING = os.environ["CONNECTION_STRING"]
# CONTAINER_NAME= os.environ["CONTAINER_NAME"]
# CONTAINER_CLIENT = ContainerClient.from_connection_string(conn_str=CONNECTION_STRING, container_name=CONTAINER_NAME)


# def ingest(file_path: str, blob_path: str):
#     try:
#         _BLOB_CLIENT = CONTAINER_CLIENT.get_blob_client(blob=blob_path)
#         with open(file_path, "rb") as data:
#             _BLOB_CLIENT.upload_blob(data=data, overwrite=True)
#     except Exception as err:
#         print("/"*10  + f" {err} or, \n {blob_path} already exists azure-blob-storage " + "\\"*10)


def generate_leo_img(prompt: str):
    with st.spinner("Wait generating the repsonse"):
        ID= str(uuid.uuid4().hex)
        #image generation
        url = "https://cloud.leonardo.ai/api/rest/v1/generations"
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {LEO_IMG_TOKEN}",
            "content-type": "application/json"
        }
        # Define the data payload
        payload = {
            "height": 512,
            "prompt": prompt,
            "modelId": LEO_MODEL_ID,  # Leonardo Phoenix Model
            "width": 512,
            "alchemy": True,
            "contrast": 4,
            "styleUUID": LEO_STYLE_UUID,
            "public": False,
            "num_images": 1,
            "enhancePrompt": False
        }
        response = requests.post(url, headers=headers, json=payload)
        time.sleep(2) 

        generation_id = response.json()["sdGenerationJob"]["generationId"]
    
        # get the generated image
        url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
        flag = True
        while flag:
            try:
                response = requests.get(url, headers=headers)
                time.sleep(5)
                image_url = response.json()["generations_by_pk"]["generated_images"][0]["url"]
                flag = False
            except IndexError as e:
                print("wait")
            except Exception as e:
                print("wait")
                flag = False        

        im = Image.open(requests.get(image_url, stream=True).raw)
        im.save(f"{ID}_Leo.png")
        st.image(image=f"{ID}_Leo.png")
        blob_path = f"OpenAI-Assistant/{ID}_Leo.png"
        # ingest(file_path=f"{ID}_Leo.png", blob_path=blob_path)
        if os.path.exists(f"{ID}_Leo.png"):
            os.remove(f"{ID}_Leo.png")
            
        return blob_path

openai_llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,  # if you prefer to pass api key in directly instaed of using env vars
)

claude_llm = ChatBedrock(
    model_id=aws_model_id,
    temperature=0,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_access_key_secret,
    region_name=aws_region
    
)

azure_llm = AzureChatOpenAI(
    azure_deployment=azure_deployment,  # or your deployment
    api_version=azure_api_version,  # or your api version
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key=azure_openai_key,
    max_retries=2,
    azure_endpoint=azure_endpoint
    # other params...
)


@tool
def multiply_numbers(num1: int, num2: int) -> int:
    """
        multiplication of numbers
    """
    return num1 * num2


@tool
def generate_image(prompt: str) -> str:
    """
        generate an image of given prompt
    """
    return generate_leo_img(prompt)

@tool
def get_relevant_docs(query, threshold=0.2):
    """
        User will ask query about document covered in RAG. 
        Respond each question like a document indexed in RAG.
    """
    # needs a local faiss_db folder with FAISS index for it to work

    faiss_db= FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
    docs_and_scores = faiss_db.similarity_search_with_score(query)
    relevant_docs = [doc for doc, score in docs_and_scores if score >= threshold]
    if not relevant_docs:
        return "The document does not cover this topic."
    context = "\n".join([doc.page_content for doc in relevant_docs])
    return context

def call_chain(model, question):
    messages = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the questions and alway give image path when user ask to generate an image. For python_tool always write a python code for it to produce output ans store any generated documents into result folder if not present create one."),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
        ])
    # chain = messages | model

    tools = [multiply_numbers, generate_image, get_relevant_docs, python_tool]

    agent = create_tool_calling_agent(model, tools, messages)
    print(chat_history_for_chain)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    ai_msg= agent_executor.invoke({"input": question, "chat_history":chat_history_for_chain.messages})
    chat_history_for_chain.add_user_message(question)
    if model== claude_llm:
        chat_history_for_chain.add_ai_message(ai_msg['output'][0]['text'])
        return ai_msg['output'][0]['text']
    else:
        chat_history_for_chain.add_ai_message(ai_msg['output'])
        return ai_msg['output']
