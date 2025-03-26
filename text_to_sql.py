# Text-to-SQL Agents in Relational Databases with CrewAI

import os
import numpy as np
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine, text
from crewai.tools import tool
from crewai import Agent, Crew, Process, Task, agent
# from crewai.crew import CrewBase
import re
from langchain_openai import ChatOpenAI
from typing import ClassVar 
from mysql.connector import Error
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Connect to the Mysql database
def connect_to_db() -> mysql.connector.MySQLConnection:
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='password',
            database="Gaya.ai"

        )
        if connection.is_connected():
            print(f"Connected to MySQL {connection.database}")
            return connection
    except Error as e:
        print("Error while connecting to MySQL", e)


# Connect to the Mysql database
__db_url = f'mysql+pymysql://root:password@localhost:3306/Gaya.ai'
ENGINE = create_engine(__db_url)

# Define OpenAI API key
#OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# OPENAI_API_KEY="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"

# create our tool to query the database
OPENAI_API_KEY=""

# @tool("execute_database")
def execute_database_query(query: str) -> str:
     """Query the Gaya.ai MySql database and return the results as a string.

    Args:
        query (str): The SQL query to execute.

    Returns:
        str: The results of the query as a string, where each row is separated by a newline.

    Raises:
        Exception: If the query is invalid or encounters an exception during execution.
    """
     with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'

        max_result_len = 1000
        ret = '\n'.join(", ".join(map(str, result)) for result in res)
        if len(ret) > max_result_len:
            ret = ret[:max_result_len] + '...\n(results too long. Output truncated.)'
        return ret

# @CrewBase
class MacSqlCrew(Crew):

    # Load the files from the config directory
    # agents_config = 'agents.yaml'
    # tasks_config = 'tasks.yaml'
    agents_config: ClassVar[str] = 'agents.yaml'
    tasks_config: ClassVar[str] = 'tasks.yaml'

    def __init__(self, **kwargs):
        # Initialize with OpenAI model
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name="gpt-4-turbo",
            temperature=0
        )
        super().__init__(llm=llm, **kwargs)
    
    '''
    @agent
    def selector_agent(self) -> Agent:
        a = Agent(
            config=self.agents_config['selector_agent'],
            llm=self.llm,
            allow_delegation=False,
            verbose=True,
            tools=[
                execute_database_query, 
            ]
        )
        return a
    '''

@tool('get_tables_from_database')
def get_tables_from_database_tool() -> str:
    """
    Retrieves a list of table names from the public schema of the connected database.

    Returns:
        str: A string containing a list of table names, each on a new line in the format:
             (Table: table_name)
             If an error occurs during execution, it returns an error message instead.

    Raises:
        Exception: If there's an error executing the SQL query, the exception is caught
                   and returned as a string message.
    """
    
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema='Gaya.ai';"
    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'
        
    tables = []
    for table in res:
        table = re.findall(r'[a-zA-Z]+', str(table))[0]
        tables.append(f'(Table: {table})\n')
    return ''.join(tables)

def _get_tables_from_database() -> str:
    """
    Retrieves a list of table names from the database.
    """
    query = "SELECT table_name FROM information_schema.tables WHERE table_schema='Gaya.ai';"
    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'
        
    tables = []
    for table in res:
        table = re.findall(r'[a-zA-Z]+', str(table))[0]
        tables.append(f'(Table: {table})\n')
    return ''.join(tables)

# Use this function in your main code
get_tables_from_database = _get_tables_from_database

@tool('get_schema_of_given_table')
def get_schema_of_given_table(
    table_name: str
) -> str:
    """
    Retrieves the schema information for a given table from the database.

    Args:
        table_name (str): The name of the table for which to retrieve the schema information.

    Returns:
        str: A string containing the schema information, with each column on a new line in the format:
             (Column: column_name, Data Type: data_type)
             If an error occurs during execution, it returns an error message instead.
    """

    query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}'"
    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'
    
    columns = []
    for column, data_type in res:
        columns.append(f'(Column: {column}, Data Type:{data_type})\n')
    return ''.join(columns)


@tool('get_distinct_column_values')
def get_distinct_column_values(
    table_name: str,
    column: str
) -> str:
    """
    Retrieves distinct values from a specified column in a given table.

    Args:
        table_name (str): The name of the table to query.
        column (str): The name of the column to retrieve distinct values from.

    Returns:
        str: A string containing the distinct values, each formatted as "(Value: <value>)\n".
             If an error occurs during query execution, it returns an error message.
    """
    
    query = f'SELECT DISTINCT {column} FROM {table_name};'
    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
        except Exception as e:
            return f'Wrong query, encountered exception {e}.'
    
    values = []
    for value in res:
        values.append(f'(Value: {value})\n')
    return ''.join(values)

def get_table_schema(table_name: str) -> str:
    """Get the schema of a given table"""
    query = f"DESCRIBE {table_name};"
    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
            columns = []
            for row in res:
                columns.append(f"Column: {row[0]}, Type: {row[1]}")
            return "\n".join(columns)
        except Exception as e:
            return f'Error getting schema: {e}'

def query_database(query: str) -> str:
    """Execute a database query directly"""
    with ENGINE.connect() as connection:
        try:
            res = connection.execute(text(query))
            results = []
            for row in res:
                results.append(", ".join(str(value) for value in row))
            return "\n".join(results)
        except Exception as e:
            return f'Query error: {e}'

def simple_ask_database_question(question: str) -> str:
    """A simpler approach that uses LangChain directly without CrewAI"""
    
    # First get database tables
    tables = get_tables_from_database()
    
    # Get schema for ImageData table
    schema = get_table_schema('ImageData')
    print(f"\nTable Schema:\n{schema}")
    
    # Create a prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert SQL database analyst. I need you to help answer a question about my database.
    
    The database has the following tables:
    {tables}
    
    The ImageData table has this schema:
    {schema}
    
    User's question: {question}
    
    First, think about which columns from the schema can help answer this question.
    Then, craft an SQL query using only the columns that exist in the schema.
    Finally, execute the query and explain the answer in plain language.
    """)
    
    # Create a chain
    chain = (
        prompt 
        | ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4-turbo") 
        | StrOutputParser()
    )
    
    # Run the chain
    suggested_query_and_analysis = chain.invoke({
        "tables": tables,
        "schema": schema,
        "question": question
    })
    
    # Try to extract the SQL query from the response
    sql_match = re.search(r"```sql\s+(.*?)\s+```", suggested_query_and_analysis, re.DOTALL)
    
    if sql_match:
        sql_query = sql_match.group(1).strip()
        print(f"Executing SQL query: {sql_query}")
        
        # Execute the query
        try:
            results = query_database(sql_query)
            return f"Results: {results}\n\nAnalysis: {suggested_query_and_analysis}"
        except Exception as e:
            return f"Error executing query: {e}\n\nModel suggestion: {suggested_query_and_analysis}"
    else:
        # If no SQL code block found, look for any query-like statements
        sql_lines = []
        for line in suggested_query_and_analysis.split('\n'):
            if 'SELECT' in line.upper() and ('FROM' in line.upper() or 'WHERE' in line.upper()):
                sql_lines.append(line)
        
        if sql_lines:
            sql_query = ' '.join(sql_lines)
            print(f"Extracted SQL query: {sql_query}")
            try:
                results = query_database(sql_query)
                return f"Results: {results}\n\nAnalysis: {suggested_query_and_analysis}"
            except Exception as e:
                return f"Error executing extracted query: {e}\n\nModel suggestion: {suggested_query_and_analysis}"
        
        return f"No SQL query could be extracted from the model's response. Here's the analysis:\n\n{suggested_query_and_analysis}"

if __name__ == "__main__":
    # First check database connection
    db_connection = connect_to_db()
    if db_connection:
        print("Database connection established successfully.")
        db_connection.close()
        
        # Example 1: List all tables
        print("\nListing all tables in the database:")
        tables = get_tables_from_database()
        print(tables)
        
        # Example 2: Ask a question
        question = "How many records are in the 'ImageData' table in the Gaya.ai database?"
        print(f"\nQuestion: {question}")
        answer = simple_ask_database_question(question)
        print(f"Answer: {answer}")
        
        # Example 3: More complex question
        question = "What are the top 5 most common values in the 'ImageData' table in the Gaya.ai database?"
        print(f"\nQuestion: {question}")
        answer = simple_ask_database_question(question)
        print(f"Answer: {answer}")

        while True:
        
            user_question = input("\nEnter your question about the database: ")
            
            # Check if user wants to exit
            if user_question.lower() in ['exit', 'quit', 'q']:
                print("Exiting the program. Goodbye!")
                break
            
            # Process the question
            print(f"\nProcessing question: {user_question}")
            try:
                answer = simple_ask_database_question(user_question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error processing question: {e}")