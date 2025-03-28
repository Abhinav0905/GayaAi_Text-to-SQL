# Text-to-SQL using LangChain
import os
import numpy as np
import mysql.connector
import pandas as pd
from sqlalchemy import create_engine, text
import re
import streamlit as st
from langchain_openai import ChatOpenAI
from typing import ClassVar 
from mysql.connector import Error
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Get database connection parameters from environment variables with fallbacks
DB_HOST = os.environ.get('DB_HOST', 'database-1-instance-1.ctg4uio0ox0p.us-east-2.rds.amazonaws.com')
DB_USER = os.environ.get('DB_USER', 'admin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'password')
DB_NAME = os.environ.get('DB_NAME', 'Gaya')
DB_PORT = os.environ.get('DB_PORT', '3306')

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'your_openai_api_key_here')

# Connect to the MySQL database
def connect_to_db() -> mysql.connector.MySQLConnection:
    try:
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        if connection.is_connected():
            print(f"Connected to MySQL {connection.database}")
            return connection
    except Error as e:
        print("Error while connecting to MySQL", e)
        return None

# Create SQLAlchemy engine
__db_url = f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}'
ENGINE = create_engine(__db_url)

def get_tables_from_database() -> str:
    """
    Retrieves a list of table names from the database.
    """
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema='{DB_NAME}';"
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

def get_schema_of_given_table(table_name: str) -> str:
    """
    Retrieves the schema information for a given table from the database.
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

def get_distinct_column_values(table_name: str, column: str) -> str:
    """
    Retrieves distinct values from a specified column in a given table.
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
    
    # Create a prompt to determine which table to use
    table_selection_prompt = ChatPromptTemplate.from_template("""
    You are an expert SQL database analyst. I need you to identify the most relevant table for this question.
    
    The database has the following tables:
    {tables}
    
    User's question: {question}
    
    Return only the name of the most relevant table, with no additional text or explanation.
    """)
    
    # Create a chain to select the table
    table_selection_chain = (
        table_selection_prompt 
        | ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4-turbo") 
        | StrOutputParser()
    )
    
    # Run the chain to get the relevant table name
    try:
        relevant_table = table_selection_chain.invoke({
            "tables": tables, 
            "question": question
        }).strip()
        
        # Clean up the table name (remove any extra characters, quotes, etc.)
        relevant_table = re.sub(r'[^a-zA-Z0-9_]', '', relevant_table)
        print(f"Selected table: {relevant_table}")
        
        # Get schema for the selected table
        schema = get_table_schema(relevant_table)
        print(f"\nTable Schema:\n{schema}")
    except Exception as e:
        # Fallback to ImageData if there's an error determining the table
        relevant_table = 'ImageData'
        schema = get_table_schema(relevant_table)
        print(f"Error determining relevant table: {e}. Falling back to {relevant_table}")
    
    # Create a prompt
    prompt = ChatPromptTemplate.from_template("""
    You are an expert SQL database analyst. I need you to help answer a question about my database.
    
    The database has the following tables:
    {tables}
    
    The {table_name} table has this schema:
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
        "table_name": relevant_table,
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

# Streamlit UI
def main():
    st.set_page_config(page_title="Gaya.ai - Text to SQL", layout="wide")
    
    st.title("Welcome to Gaya.ai")
    st.header("Ask Your Questions")
    st.markdown("This is a simple text-to-SQL agent that can help you query the Gaya.ai database.")
    
    # Display connection info (excluding password)
    if st.sidebar.checkbox("Show Connection Info", False):
        st.sidebar.write(f"Database Host: {DB_HOST}")
        st.sidebar.write(f"Database Name: {DB_NAME}")
        st.sidebar.write(f"Database User: {DB_USER}")
        st.sidebar.write(f"Database Port: {DB_PORT}")
    
    # Check database connection
    try:
        db_connection = connect_to_db()
        if not db_connection:
            st.error("⚠️ Could not connect to the database. Please check your database connection parameters.")
            st.info("Make sure your EC2 security group allows connections to the RDS database.")
            return
        db_connection.close()
        st.success("✅ Connected to database successfully!")
    except Exception as e:
        st.error(f"⚠️ Database connection error: {e}")
        st.info("Please check your database connection parameters and network settings.")
        return
    
    user_question = st.text_input("Enter your question here:")
    
    if st.button("Submit", key="submit_button"):
        if not user_question:
            st.warning("Please enter a question.")
            return
            
        with st.spinner("Processing your question..."):
            try:
                answer = simple_ask_database_question(user_question)
                st.write("### Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error processing question: {e}")
                st.info("This might be due to API key issues or database connectivity problems.")

if __name__ == "__main__":
    main()





