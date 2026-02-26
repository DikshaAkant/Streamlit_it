#Streamlit + langchain + ollama
import os
import streamlit as st

# Import ollama LLM from langchain community package
from langchain_community.llms import Ollama

#Import prompt template to structure input
from langchain_core.prompts import ChatPromptTemplate

#Import output parser to convert model output into string
from langchain_core.output_parsers import StrOutputParser

#Step 1: Creates prompt templates
# This defines how the AI should behave and how it recieves user input 
prompt = ChatPromptTemplate.from_messages(
    [
        # System message defines AI behaviour
        ("system", "You are a helpful assistant. Please respond."),
        
        # uSer message contains placeholder {question}
        ("human", "Question: {question}")
    ]
)


#Step 2: Streamlit App UI
#App Title
st.title("Streamlit : Langchain with gemma model(ollama)")
# Text input box for user question
input_text = st.text_input("What question do u have in your mind")

#step 3:Load Ollama Model
#Load local gemma model 
llm = Ollama(model="gemma2:2b")
# Convert model output to string 
output_parser = StrOutputParser()

# Creates langchain pipeline (Prompts -> Model -> Output parser)
chain = prompt | llm  | output_parser

# Run model when user inputs questions
if(input_text):
    response = chain.invoke({"question: ",input_text})
    st.write(response)