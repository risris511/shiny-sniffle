import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Set environment variable for Hugging Face API token

load_dotenv()
api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load documents
@st.cache_resource
def load_documents():
    loader = PyPDFDirectoryLoader("uh")
    docs_before_split = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    return text_splitter.split_documents(docs_before_split)

docs_after_split = load_documents()

# Create embeddings
@st.cache_resource
def create_vectorstore(_docs):
    huggingface_embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    return FAISS.from_documents(_docs, huggingface_embeddings)

vectorstore = create_vectorstore(docs_after_split)

# Set up the T5 model and tokenizer
model_id = "risris8787/rag"
tokenizer = T5Tokenizer.from_pretrained(model_id)
model = T5ForConditionalGeneration.from_pretrained(model_id)

# Use HuggingFacePipeline for text-to-text generation
hf_pipeline = HuggingFacePipeline.from_model_id(
    model_id=model_id,
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 300}
)

# Streamlit application layout
st.title("Document Retrieval and QA System")
st.write("Enter your question below:")

# User input for question
question = st.text_input("Question", value="Who is the customer with most items?")

if st.button("Get Answer"):
    # Query the LLM directly (optional)
    response_pipeline = hf_pipeline.invoke(question)
    st.write("Response from HuggingFacePipeline:", response_pipeline)

    # Set up prompt template
    prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
    2. If you find the answer, write the answer in a concise way with five sentences maximum.

    {context}

    Question: {question}
    Helpful Answer:
    """

    PROMPT = PromptTemplate(input_variables=["context", "question"],
                            template=prompt_template)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    retrievalQA = RetrievalQA.from_chain_type(
        llm=hf_pipeline,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    # Invoke the retrieval QA
    result = retrievalQA.invoke({"query": question})

    # Display results
    st.write("Answer:", result['result'])

    relevant_docs = result['source_documents']
    st.write(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')
    st.write("*" * 100)

    for i, doc in enumerate(relevant_docs):
        st.write(f"Relevant Document #{i + 1}:")
        st.write(f"Source file: {doc.metadata['source']}, Page: {doc.metadata['page']}")
        st.write(f"Content: {doc.page_content}")
        st.write("-" * 100)
