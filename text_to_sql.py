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
        return None

# Connect to the Mysql database
__db_url = f'mysql+pymysql://root:password@localhost:3306/Gaya.ai'
ENGINE = create_engine(__db_url)

# Define OpenAI API key
OPENAI_API_KEY="your_openai_api_key_here"
def get_tables_from_database() -> str:
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

# Streamlit UI
def main():
    st.set_page_config(page_title="Gaya.ai - Text to SQL", layout="wide")
    
    st.title("Welcome to Gaya.ai")
    st.header("Ask Your Questions")
    st.markdown("This is a simple text-to-SQL agent that can help you query the Gaya.ai database.")
    
    # Check database connection
    db_connection = connect_to_db()
    if not db_connection:
        st.error("⚠️ Could not connect to the database. Please check your database connection.")
        return
    db_connection.close()
    
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

if __name__ == "__main__":
    main()





