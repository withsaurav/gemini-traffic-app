import os
import json
from typing import Optional
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

from google.cloud import bigquery
from google.oauth2 import service_account


class BigQueryWrapper:
    def __init__(self, project_id: str, service_account_path: Optional[str] = None, credentials=None):
        if credentials:
            self.client = bigquery.Client(credentials=credentials, project=project_id)
        elif service_account_path:
            creds = service_account.Credentials.from_service_account_file(service_account_path)
            self.client = bigquery.Client(credentials=creds, project=project_id)
        else:
            raise ValueError("Either credentials or service_account_path must be provided")

        self.project_id = project_id

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

    def run(self, sql: str) -> str:
        query_job = self.client.query(sql)
        results = query_job.result()
        output = []
        for row in results:
            output.append(dict(row))
        return json.dumps(output, indent=2)


class BigQueryAgent:
    def __init__(self, service_account_path: Optional[str], project_id: str, gemini_api_key: str, dataset_id: str, credentials=None):
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

        if credentials:
            self.bq = BigQueryWrapper(project_id=project_id, credentials=credentials)
        else:
            self.bq = BigQueryWrapper(project_id=project_id, service_account_path=service_account_path)

        self.dataset_id = dataset_id
        self.project_id = project_id
        self.schema_info = self.bq.get_schema_info(dataset_id)

        def query_bigquery(sql: str) -> str:
            return self.bq.run(sql)

        self.tool = Tool(
            name="BigQuery_SQL",
            func=query_bigquery,
            description=f"""Query BigQuery dataset `{project_id}.{dataset_id}`.
Available schema:
{self.schema_info}

IMPORTANT SQL RULES:
1. Always use backticks around table names: `{project_id}.{dataset_id}.table_name`
2. Use LIMIT to avoid large result sets (max 1000 rows)
3. For date operations, use PARSE_DATE or SAFE_CAST functions
4. Write clean, valid BigQuery SQL syntax
"""
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            convert_system_message_to_human=False,
            google_api_key=gemini_api_key
        )

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
            enhanced_question = f"""
You are a data analyst assistant working with BigQuery.

Dataset location: `{self.project_id}.{self.dataset_id}`

Available tables and schema:
{self.schema_info}

Please answer the following question by writing a valid BigQuery SQL query:
{question}

Follow these SQL rules:
1. Always use backticks around full table names like `project.dataset.table`.
2. Use LIMIT in large queries.
3. Handle string/date fields properly.
4. If unsure about the question, assume user refers to the only available table.
"""
            return self.agent.run(enhanced_question)
        except Exception as e:
            return f"Error: {str(e)}"

    def run_sql_directly(self, sql: str) -> str:
        return self.bq.run(sql)
