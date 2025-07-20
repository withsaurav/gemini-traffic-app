import os
import json
from typing import Optional
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI

from google.cloud import bigquery
from google.oauth2 import service_account
import re



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

    def clean_sql_query(self, sql: str) -> str:
        """Clean and validate SQL query"""
        # Remove any markdown formatting that might have been added
        sql = re.sub(r'```sql\n?', '', sql)
        sql = re.sub(r'```\n?', '', sql)
        
        # Remove any extra whitespace and newlines
        sql = sql.strip()
        
        # Replace any problematic characters
        sql = sql.replace('`', '`')  # Replace any fancy backticks with regular ones
        sql = sql.replace(''', "'")  # Replace smart single quotes
        sql = sql.replace(''', "'")  # Replace smart single quotes (closing)
        sql = sql.replace('"', '"')  # Replace smart double quotes (opening)
        sql = sql.replace('"', '"')  # Replace smart double quotes (closing)
        
        return sql

    def get_schema_info(self, dataset_id: str) -> str:
        """Get schema information for the dataset"""
        try:
            dataset_ref = self.client.dataset(dataset_id)
            tables = list(self.client.list_tables(dataset_ref))

            schema_info = f"Dataset: {self.project_id}.{dataset_id}\n\nAvailable Tables:\n"

            for table in tables:
                table_ref = dataset_ref.table(table.table_id)
                table_obj = self.client.get_table(table_ref)

                schema_info += f"\n{table.table_id} ({table_obj.num_rows} rows):\n"
                for field in table_obj.schema:
                    schema_info += f"  - {field.name} ({field.field_type})\n"

            return schema_info

        except Exception as e:
            return f"Error getting schema: {str(e)}"

  #  def run(self, sql: str) -> str:
  #      """Execute SQL query and return results as string"""
  #      try:
  #          # Clean the SQL query
  #          cleaned_sql = self.clean_sql_query(sql)
            
  #          print(f"Executing SQL: {cleaned_sql}")  # Debug print
            
  #          query_job = self.client.query(cleaned_sql)
  #          results = query_job.result()

  #          # Convert to pandas DataFrame for better formatting
  #          df = results.to_dataframe()

  #          if df.empty:
  #              return "No results found."

            # Return formatted results
  #          result_str = f"Query returned {len(df)} rows:\n\n"
  #          result_str += df.to_string(index=False, max_rows=50)

            # Add summary statistics for numeric columns
  #          numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
  #          if len(numeric_cols) > 0:
  #              result_str += "\n\nSummary Statistics:\n"
  #              for col in numeric_cols:
  #                  result_str += f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}\n"

  #          return result_str

  #      except Exception as e:
  #          return f"Error executing query: {str(e)}\nSQL attempted: {sql}"



    def run(self, sql: str, return_df: bool = False) -> Union[str, pd.DataFrame]:
        """Execute SQL query and return results as string or DataFrame"""
        try:
            # Clean the SQL query
            cleaned_sql = self.clean_sql_query(sql)
    
            print(f"Executing SQL: {cleaned_sql}")  # Debug print
    
            query_job = self.client.query(cleaned_sql)
            df = query_job.result().to_dataframe()
    
            if return_df:
                return df  # ✅ Return DataFrame if required
    
            if df.empty:
                return "No results found."
    
            # Return formatted results as string
            result_str = f"Query returned {len(df)} rows:\n\n"
            result_str += df.to_string(index=False, max_rows=50)
    
            # Add summary stats if any numeric columns exist
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                result_str += "\n\nSummary Statistics:\n"
                for col in numeric_cols:
                    result_str += f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}\n"
    
            return result_str
    
        except Exception as e:
            return f"Error executing query: {str(e)}\nSQL attempted: {sql}"



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

        # Initialize LangChain tool
        #self.tool = self._create_tool()
        self.tool = Tool(
            name="BigQuery_SQL",
            func=query_bigquery,
            description=(
                f"Execute SQL queries on BigQuery dataset `{self.project_id}.{self.dataset_id}`.\n\n"
                f"📘 Available schema:\n{self.schema_info}\n\n"
                f"💡 SQL Guidelines:\n"
                f"- Use backticks around table names\n"
                f"- Always use LIMIT to avoid large results\n"
                f"- Handle date/time fields with SAFE_CAST or PARSE_DATE\n"
                f"- Use LOWER(column_name) for case-insensitive string filtering, e.g., `WHERE LOWER(color) = 'silver'`\n"
                f"- Write clean, valid BigQuery SQL syntax\n"
            )
        )

        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            convert_system_message_to_human=False,
            google_api_key=gemini_api_key
        )


        # Initialize LangChain agent
        self.agent = initialize_agent(
            tools=[self.tool],
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            #agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=2 )
        

        
        

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
                #return self.agent.run(enhanced_question)
                response = self.agent.run(enhanced_question)

               
               # Check if the agent returned a SQL query string
                if response.strip().lower().startswith("select"):
                    return self.bq.run(response, return_df=True)
                    #df = self.bq.run(response)
                    #return df  # Return actual query result
                return response
                
            except Exception as e:
                return f"Error: {str(e)}"
        

    def run_sql_directly(self, sql: str) -> str:
            """Run raw SQL against BigQuery directly."""
            return self.bq.run(sql)
        

        
