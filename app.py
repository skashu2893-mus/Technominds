import streamlit as st
import fitz  # PyMuPDF
import faiss
import os
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import APIClient
import tempfile
import uuid

# Initialize Watsonx AI Model (Update your credentials here)
wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "your-api-key"
}

project_id = "your-project-id"
model_id = "ibm/granite-13b-instruct-v2"  # Use "ibm-instruct-3.2-2b" if this is your model ID

client = APIClient(wml_credentials)
client.set.default_project(project_id)

model = Model(model_id=model_id, params={GenParams.MAX_NEW_TOKENS: 300}, client=client)

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Streamlit UI
st.set_page_config(page_title="StudyMate", layout="_
