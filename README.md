# Multiagent chatbot

The chatbot can perform various operations:
1. Answer questions from the documents indexed in the RAG.
2. Answer question and perform statistical analysis and generate plots on the data file uploaded.
3. Generate the image using Leonardo.AI


The agent selects which operation to perform based on the user input and then the corresponding operation is implemented. 

![vid](static/dummy.gif)

## Dependencies
dependencies are all mentioned in `requirements.txt` the project is built using `Python3.12`

## Key requirements
Because the agent also allows multiple LLM models it needs keys for all the models being made available by agent, models currently offered by the agent:
1. Azure OpenAI GPT-4o
2. OpenAI GPT-4o
3. AWS Bedrock Haiku

Other than credentials for these LLM agents, the module also needs `faiss-db` created using the script `create_db.py` to generate a database for RAG to answer questions from the document indexed in `create_db.py`.

Alongside LLM credentials the model also need HuggingFace token to access `Llama-3.2` model and `Leonardo.AI` keys to generate the models.

## Environment Variables

### Azure blob storage credentials to upload the generated Leonardo.AI images in
- CONNECTION_STRING
- CONTAINER_NAME

### Leonardo.AI keys to generate images
- LEO_IMG_TOKEN
- LEO_MODEL_ID
- LEO_MODEL_ID
- LEO_STYLE_UUID

### Huggingface token to access Llama-3.2 model
- HUGGINGFACEHUB_API_TOKEN

### OpenAI key to use OpenAI agent
- OPENAI_KEY

### AWS secrets to access Bedrock Haiku, and Sonnet models
- AWS_ACCESS_KEY_ID
- AWS_ACCESS_KEY_SECRET
- AWS_REGION
- AWS_MODEL_ID

### Azure OpenAI credentials to access Azure OpenAI models
- AZURE_DEPLOYMENT
- AZURE_MODEL_VERSION
- AZURE_OPENAI_KEY
- AZURE_ENDPOINT