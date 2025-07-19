import os
import re
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent, AgentType

class BigQueryWrapper:
    def __init__(self, service_account_path: str, project_id: str):
        credentials = service_account.Credentials.from_service_account_file(service_account_path)
        self.client = bigquery.Client(credentials=credentials, project=project_id)
        self.project_id = project_id

    def clean_sql_query(self, sql: str) -> str:
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        sql = sql.strip()
        return sql

    def run(self, sql: str) -> str:
        cleaned_sql = self.clean_sql_query(sql)
        query_job = self.client.query(cleaned_sql)
        results = query_job.result()
        df = results.to_dataframe()
        if df.empty:
            return "No results found."
        return df.to_string(index=False, max_rows=50)

    def get_schema_info(self, dataset_id: str) -> str:
        schema_info = ""
        dataset_ref = self.client.dataset(dataset_id)
        tables = list(self.client.list_tables(dataset_ref))
        for table in tables:
            table_ref = dataset_ref.table(table.table_id)
            schema = self.client.get_table(table_ref).schema
            schema_info += f"{table.table_id}:\n"
            for field in schema:
                schema_info += f"  - {field.name} ({field.field_type})\n"
        return schema_info.strip()


class BigQueryAgent:
    def __init__(self, service_account_path: str, project_id: str, gemini_api_key: str, dataset_id: str):
        os.environ["GOOGLE_API_KEY"] = gemini_api_key
        self.bq = BigQueryWrapper(service_account_path, project_id)
        self.dataset_id = dataset_id
        self.project_id = project_id
        self.schema_info = self.bq.get_schema_info(dataset_id)

        # Function to query BigQuery
        def query_bigquery(sql: str) -> str:
            return self.bq.run(sql)

        # Tool definition for LangChain agent
        self.tool = Tool(
            name="BigQuery_SQL",
            func=query_bigquery,
            description=f"""Query BigQuery dataset `{project_id}.{dataset_id}`.
            Available schema:
            {self.schema_info}"""
        )

        # LLM configuration
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            convert_system_message_to_human=False,
            google_api_key=gemini_api_key
        )

         # LangChain agent setup
        self.agent = initialize_agent(
            tools=[self.tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2
        )

    def ask(self, question: str) -> str:
        try:
            return self.agent.run(question)
        except Exception as e:
            return f"Error: {str(e)}"