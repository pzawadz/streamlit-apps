import time

import boto3
import streamlit as st
from langchain.chains import ConversationChain
from langchain.llms.bedrock import Bedrock
from langchain.memory import ConversationBufferMemory

st.title("ChatBot based on Amazon Bedrock using Langchain chains and Streamlit frontend")
# streamlit subtitle and radio buttons with 4 model options to select what model to use and save choice into `modeltype` 
st.subheader("Select a model to use")
modeltype = st.radio("Model type", ("Cloude V2.1", "Llama2", "Cohere Command", "Titan"))
st.write("You selected:", modeltype)

# Setup bedrock
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
    )



# @st.cache_resource
def load_llm(modeltype):
    ##llm = Bedrock(client=bedrock_runtime, model_id="anthropic.claude-v2:1")
    ##llm.model_kwargs = {"temperature": 0.7, "max_tokens_to_sample": 4096}

    if modeltype == "Cloude V2.1":
        modelid = "anthropic.claude-v2:1"
    elif modeltype == "Llama2":
        modelid = "meta.llama2-13b-chat-v1"
    elif modeltype == "Cohere Command":
        modelid = "cohere.command-light-text-v14"
    elif modeltype == "Titan":
        modelid = "amazon.titan-text-express-v1"

    llm = Bedrock(client=bedrock_runtime, model_id=modelid)
    llm.model_kwargs = {"temperature": 0.5}


    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())

    return model


model = load_llm(modeltype)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = modeltype +  " answers:  "

        # prompt = prompt_fixer(prompt)
        result = model.predict(input=prompt)

        # Simulate stream of response with milliseconds delay
        for chunk in result.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
