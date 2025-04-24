# GayaAI: Text-to-SQL Converter

GayaAI is a Python-based application that allows users to convert natural language queries into SQL statements effortlessly. This project is designed to bridge the gap between non-technical users and databases by providing an intuitive interface for querying databases using plain English.

---

## Features

- **Natural Language to SQL Conversion**:
  - Translates natural language queries into SQL statements with accuracy and efficiency.
  
- **Multi-Language Support**:
  - Supports multiple SQL dialects for various database systems (e.g., MySQL, PostgreSQL, SQLite).

- **Machine Learning Integration**:
  - Uses state-of-the-art transformer-based models for parsing and understanding queries.

- **Interactive Notebook Support**:
  - Includes Jupyter Notebook examples for experimentation and testing.

- **Extensibility**:
  - Easily extendable to include custom SQL dialects or additional natural language processing features.

---

## Technology Stack

- **Programming Language**:
  - Python (74.1%)
  - Jupyter Notebook (25.9%)

- **Frameworks & Libraries**:
  - Natural Language Processing: `transformers`, `spacy`
  - SQL Parsing: `sqlparse`
  - Machine Learning: `torch`, `tensorflow` (if applicable)
  - Data Handling: `pandas`, `numpy`

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Abhinav0905/GayaAi_Text-to-SQL.git
   cd GayaAi_Text-to-SQL

2. Install Dependencies
   pip install -r requirements.txt

Setup Environment:
Ensure Python 3.8 or higher is installed.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Run the Application: 
python main.py

Usage
Interactive Mode:

Run the application and input natural language queries to generate SQL statements.
Jupyter Notebooks:

Open the provided .ipynb files for step-by-step examples and testing.
APIs:

Integrate the tool into your applications using the exposed APIs.
Examples
Input: "Show all users who registered in the last 7 days."
Output:
SELECT * FROM users WHERE registration_date >= NOW() - INTERVAL '7 days';

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Inspired by advancements in NLP and SQL parsing technologies.
Special thanks to the open-source community for contributing to the libraries and tools used in this project.
