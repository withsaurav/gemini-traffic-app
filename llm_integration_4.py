import os
import json
from typing import Optional
from langchain.agents import Tool, AgentType, initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from google.cloud import bigquery
from google.oauth2 import service_account
import re


class BigQueryWrapper:
    """Handles BigQuery database operations and query execution"""
    
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
        """Get comprehensive schema information for the dataset"""
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

    def execute_query(self, sql: str) -> str:
        """Execute SQL query and return formatted results"""
        try:
            # Clean the SQL query
            cleaned_sql = self.clean_sql_query(sql)
            
            print(f"Executing SQL: {cleaned_sql}")  # Debug print
            
            query_job = self.client.query(cleaned_sql)
            results = query_job.result()

            # Convert to pandas DataFrame for better formatting
            df = results.to_dataframe()

            if df.empty:
                return "No results found."

            # Return formatted results
            result_str = f"Query returned {len(df)} rows:\n\n"
            result_str += df.to_string(index=False, max_rows=50)

            # Add summary statistics for numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 0:
                result_str += "\n\nSummary Statistics:\n"
                for col in numeric_cols:
                    result_str += f"{col}: min={df[col].min()}, max={df[col].max()}, avg={df[col].mean():.2f}\n"

            return result_str

        except Exception as e:
            return f"Error executing query: {str(e)}\nSQL attempted: {sql}"


class BigQueryAgent:
    """
    Intelligent BigQuery data assistant that provides natural language responses to database queries.
    Uses a two-step approach: SQL generation and natural language response formatting.
    """
    
    def __init__(self, service_account_path: Optional[str], project_id: str, gemini_api_key: str, dataset_id: str, credentials=None):
        """
        Initialize the BigQuery Agent
        
        Args:
            service_account_path: Path to Google Cloud service account JSON file
            project_id: Google Cloud project ID
            gemini_api_key: Google Gemini API key
            dataset_id: BigQuery dataset ID
            credentials: Optional Google Cloud credentials object
        """
        os.environ["GOOGLE_API_KEY"] = gemini_api_key

        if credentials:
            self.bq = BigQueryWrapper(project_id=project_id, credentials=credentials)
        else:
            self.bq = BigQueryWrapper(project_id=project_id, service_account_path=service_account_path)

        self.dataset_id = dataset_id
        self.project_id = project_id
        self.schema_info = self.bq.get_schema_info(dataset_id)

        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            convert_system_message_to_human=False,
            google_api_key=gemini_api_key
        )

        # Create memory to store conversation context
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Initialize SQL execution tool
        self._setup_sql_tool()
        
        # Initialize SQL generation agent
        self._setup_sql_agent()
        
        # Initialize response generation chain
        self._setup_response_generator()

    def _setup_sql_tool(self):
        """Setup the SQL execution tool with comprehensive guidelines"""
        def execute_bigquery_sql(sql: str) -> str:
            return self.bq.execute_query(sql)

        self.sql_tool = Tool(
            name="BigQuery_SQL_Executor",
            func=execute_bigquery_sql,
            description=(
                f"Execute SQL queries on BigQuery dataset `{self.project_id}.{self.dataset_id}`.\n"
                f"Available schema:\n{self.schema_info}\n\n"
                f"SQL Guidelines:\n"
                f"- Use backticks around table names: `{self.project_id}.{self.dataset_id}.table_name`\n"
                f"- Always use LIMIT to avoid large results\n"
                f"- For string filtering, ALWAYS use LOWER() for case-insensitive matching\n"
                f"- Example: WHERE LOWER(color_column) = LOWER('Blue') or WHERE LOWER(color_column) = 'blue'\n"
                f"- Handle date/time fields with SAFE_CAST or PARSE_DATE\n"
                f"- For partial string matches, use LIKE with LOWER(): WHERE LOWER(column) LIKE '%blue%'\n"
                f"- Always check for both exact matches and partial matches when filtering strings"
            )
        )

    def _setup_sql_agent(self):
        """Setup the SQL generation and execution agent"""
        self.sql_agent = initialize_agent(
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            tools=[self.sql_tool],
            llm=self.llm,
            verbose=True,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=2
        )

    def _setup_response_generator(self):
        """Setup the natural language response generator"""
        self.response_prompt = PromptTemplate(
            input_variables=["user_question", "query_results", "schema_info"],
            template="""
You are a helpful data analyst assistant. A user asked a question about their data, and you have the query results.

USER QUESTION: {user_question}

QUERY RESULTS: {query_results}

DATABASE SCHEMA: {schema_info}

Your task is to provide a natural, conversational response to the user's question based on the query results.

RESPONSE GUIDELINES:
- Use natural language and complete sentences
- For count queries, say "There are X [items] found in the records" or "I found X [items] in the database"
- For list queries, say "I found the following [items]..." and then list them
- Be helpful and explain what the results mean
- If showing numbers, put them in context
- Be conversational and friendly
- If no results found, explain that clearly

Provide your response:"""
        )

        self.response_generator = LLMChain(
            llm=self.llm,
            prompt=self.response_prompt
        )

    def ask(self, question: str) -> str:
        """
        Ask a question about your data and get a natural language response
        
        Args:
            question: Natural language question about the data
            
        Returns:
            Natural language response with insights from the data
        """
        return self._process_query(question)

    def _process_query(self, question: str) -> str:
        """Internal method to process queries using the two-step approach"""
        try:
            print("\n[SQL Agent] Generating and executing query...\n")
            
            # Step 1: Generate SQL and get query results
            sql_prompt = f"""
Based on this question about our database: "{question}"

Dataset: `{self.project_id}.{self.dataset_id}`
Schema: {self.schema_info}

IMPORTANT SQL GUIDELINES:
- For any color/string filtering, use LOWER() function for case-insensitive matching
- Example: WHERE LOWER(color) = 'blue' or WHERE LOWER(color) LIKE '%blue%'
- Always check the actual column names in the schema above
- Use backticks around full table names

Generate and execute the appropriate SQL query to answer this question.
Use the BigQuery_SQL_Executor tool to run the query.
"""
            query_results = self.sql_agent.invoke(sql_prompt)
            
            print("\n[Response Generator] Creating natural language response...\n")
            
            # Step 2: Generate natural language response
            response = self.response_generator.invoke({
                "user_question": question,
                "query_results": query_results,
                "schema_info": self.schema_info
            })
            
            return response['text']
            
        except Exception as e:
            return f"I encountered an error while processing your question: {str(e)}"

    def get_data_summary(self, table_name: str, column_name: str) -> str:
        """
        Get a summary of unique values in a specific column
        
        Args:
            table_name: Name of the table
            column_name: Name of the column to analyze
            
        Returns:
            Summary of unique values and their counts
        """
        try:
            summary_sql = f"""
            SELECT 
                {column_name},
                LOWER({column_name}) as lower_value,
                COUNT(*) as count
            FROM `{self.project_id}.{self.dataset_id}.{table_name}`
            GROUP BY {column_name}, LOWER({column_name})
            ORDER BY count DESC
            LIMIT 20
            """
            return self.bq.execute_query(summary_sql)
        except Exception as e:
            return f"Error getting summary: {str(e)}"

    def execute_raw_sql(self, sql: str) -> str:
        """
        Execute raw SQL query directly (for advanced users)
        
        Args:
            sql: Raw SQL query string
            
        Returns:
            Query results
        """
        return self.bq.execute_query(sql)

    def get_schema(self) -> str:
        """
        Get the database schema information
        
        Returns:
            Formatted schema information
        """
        return self.schema_info

    def get_conversation_history(self) -> str:
        """
        Get the conversation history
        
        Returns:
            Formatted conversation history
        """
        return str(self.memory.buffer)

    def clear_conversation_history(self):
        """Clear the conversation history"""
        self.memory.clear()


# Factory function for easy initialization
def create_bigquery_assistant(service_account_path: str, project_id: str, gemini_api_key: str, dataset_id: str, credentials=None) -> BigQueryAgent:
    """
    Factory function to create a BigQuery assistant
    
    Args:
        service_account_path: Path to Google Cloud service account JSON file
        project_id: Google Cloud project ID  
        gemini_api_key: Google Gemini API key
        dataset_id: BigQuery dataset ID
        credentials: Optional Google Cloud credentials object
        
    Returns:
        Configured BigQueryAgent instance
    """
    return BigQueryAgent(
        service_account_path=service_account_path,
        project_id=project_id,
        gemini_api_key=gemini_api_key,
        dataset_id=dataset_id,
        credentials=credentials
    )


# Example usage and testing
if __name__ == "__main__":
    """
    Example usage of the BigQuery Assistant
    
    # Step 1: Initialize the assistant
    assistant = create_bigquery_assistant(
        service_account_path="path/to/service-account.json",
        project_id="your-project-id",
        gemini_api_key="your-gemini-api-key", 
        dataset_id="your-dataset-id"
    )

    # Step 2: Ask questions in natural language
    print("=== Car Count Example ===")
    response = assistant.ask("how many cars are found in the records")
    print(response)  # "There are 40 cars found in the records."
    
    print("\n=== Blue Cars Example ===")
    response = assistant.ask("how many blue cars are found")
    print(response)  # "There are 15 blue cars found in the records."
    
    print("\n=== Data Summary Example ===") 
    summary = assistant.get_data_summary("cars", "color")
    print(summary)
    
    print("\n=== Schema Information ===")
    schema = assistant.get_schema()
    print(schema)
    
    # Step 3: Advanced usage
    raw_result = assistant.execute_raw_sql("SELECT COUNT(*) FROM `project.dataset.table`")
    print(raw_result)
    """
    pass