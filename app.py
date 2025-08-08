import streamlit as st
import os
import json
import pandas as pd

import os
import sys

# Ensure the current directory (where llm_integration.py is) is in Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from llm_integration_4 import BigQueryAgent


from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_info(
    json.loads(st.secrets["SERVICE_ACCOUNT_JSON"])
)
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

PROJECT_ID = "steam-airfoil-341409"
DATASET_ID = "License_Plate"

agent = BigQueryAgent(
    service_account_path=None,
    project_id=PROJECT_ID,
    gemini_api_key=os.environ["GOOGLE_API_KEY"],
    dataset_id=DATASET_ID,
    credentials=credentials  # ← Now passed correctly
)
agent.bq.client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

st.set_page_config(page_title="Gemini App", layout="centered")
st.title("🚦 Gemini App to Retrieve Traffic Data")
st.markdown("Ask a question and get SQL-backed answers using Gemini + BigQuery.")

question = st.text_input("🔍 Ask a question about traffic data:")
if st.button("Ask Question") and question:
    with st.spinner("Thinking..."):
        response = agent.ask(question)
        st.markdown("### ✅ Response")
        st.text(response)

