from crewai.tools import tool

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

# Create the tool version for agents
@tool('get_tables_from_database')
def get_tables_from_database_tool() -> str:
    """
    Retrieves a list of table names from the public schema of the connected database.
    """
    return _get_tables_from_database()