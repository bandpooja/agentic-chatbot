import os
from pathlib import Path
from langchain_openai import ChatOpenAI
import streamlit as st
from langchain_framework import call_chain, openai_llm, claude_llm, azure_llm
from langchain_openai import ChatOpenAI
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_openai import AzureChatOpenAI
import pandas as pd
import re
import time


def stream_response(response_text):
    """
        Function to stream text response
    """
    sentences = re.split(r'(?<!\w\.\w)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s', response_text)
    for sentence in sentences:
        stream = sentence.strip() + "\n"
        for word in stream.split():
            time.sleep(0.05)
            yield word+" "


if __name__=="__main__":
    dataframe= pd.DataFrame()
    # Page configuration

    st.set_page_config(page_icon=":speech_balloon:",page_title="Custom Bot")

    # Title of the application
    st.subheader("Custom Bot")

    # Initialize session state
    if "start_chat" not in st.session_state:
        st.session_state.start_chat = True
    if "messages" not in st.session_state:
        st.session_state.messages = []
    

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create two columns: one for model selection, the other for the prompt input
    _, _ = st.sidebar.columns([3,3])
    _, _ = st.sidebar.columns([3,3])
    col1, _ = st.sidebar.columns([10,3])

    with col1:
        # Model selection dropdown
        option = st.selectbox(
            "Select Model",
            ("Open AI", "Claude", "Azure Open AI")
        )
    _, _ = st.sidebar.columns([3,3])
    _, _ = st.sidebar.columns([3,3])
    _, _ = st.sidebar.columns([3,3])
    col3, _= st.sidebar.columns([30,0.5])

    with col3:
        # Text input for the user's message
        prompt = st.text_area("Enter your message")

    _, _ = st.sidebar.columns([3,3])
    _, _ = st.sidebar.columns([3,3])
    _, _ = st.sidebar.columns([3,3])
    col5, _= st.sidebar.columns([5,1])

    with col5:
        uploaded_file = st.file_uploader("Upload file", type=["csv", "php", "tex", "jpg", "docx", "js", "xml", "rb", "doc", "pdf", "xlsx", "py", "csv", "sh", "txt", "json", "c", "cs", "cpp", "jpeg", "ts", "java", "html", "md", "png", "css", "pptx"])
        if uploaded_file is not None:
            if uploaded_file.name.endswith(".csv"):
                dataframe = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xslx"):
                dataframe = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".json"):
                dataframe = pd.read_json(uploaded_file)
            elif uploaded_file.name.endswith(".html"):
                dataframe = pd.read_html(uploaded_file)
            # load content of other formats using modules
            

    _, _ = st.sidebar.columns([3,3])
    col5, _= st.sidebar.columns([3,1])

    with col5:
        # Text input for the user's message
        button= st.button("Enter")

    # Submit button placed below the columns
    if button and prompt:
        # Model assignment based on selection
        if option == "Open AI":
            model = openai_llm
        elif option == "Claude":
            model = claude_llm
        elif option == "Azure Open AI":
            model = azure_llm
        else:
            model = openai_llm  # Default model

        # Display user message
        with st.chat_message("user"):
            st.write(prompt)

        # Store user message in session state
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Get model response
        with st.spinner("Wait generating the repsonse"):
            with st.chat_message("assistant"):
                if not dataframe.empty:
                    prompt = dataframe.to_string(index=False) +"  "+ prompt
                response = call_chain(model, prompt)
                st.write(stream_response(response)) # Display the response
                image_path= os.listdir('result/')
                for img in image_path:
                    # check if the image ends with png
                    if (img.endswith(".png")):
                        st.image(image=f"result/{img}")
                for img in image_path: 
                    if os.path.isfile(f"result/{img}"):
                        os.remove(f"result/{img}")                  
                # Store assistant response in session state
        st.session_state.messages.append({"role": "assistant", "content": response})
