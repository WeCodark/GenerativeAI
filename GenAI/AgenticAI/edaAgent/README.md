# Agentic AI: Autonomous Exploratory Data Analysis (EDA)

This project creates an autonomous **Exploratory Data Analysis (EDA) Agent** using [LangChain](https://python.langchain.com/) and [Groq's](https://groq.com/) blazing-fast API. Supported by a specifically curated `prompt flow`, this agent dives directly into the `bank-additional-full.csv` marketing dataset and extracts actionable insights regarding customer demographics and campaign conversion patterns.

---

## 📋 Prerequisites

Before running the agent, make sure you have the following installed:
- **Python 3.9+** 
- **Pip** (Package manager)
- **Groq API Key** (You can grab a free API key at [Groq Console](https://console.groq.com/))

## ⚙️ Installation & Setup

1. **Navigate to your local repository**
   ```bash
   cd /Users/pranav/PRNV/AgenticAI
   ```

2. **Create and Activate a Virtual Environment (Recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Packages**
   Install the libraries contained in `requirements.txt`, which includes LangChain, LangChain-Groq, Pandas, Python-Dotenv, and Tabulate.
   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. **Set Up the Environment Variables**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_actual_api_key_here
   ```

---

## 🛠️ Step-by-Step Code Implementation

The autonomous flow happens inside `eda_agent.py` and strictly follows an analytical phase-gate structure.

### 1. Library Imports & LLM Configurations
We load the necessary environment variables safely via `dotenv`, and import the `create_pandas_dataframe_agent`. 

```python
import os
import pandas as pd
from dotenv import load_dotenv

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

load_dotenv()
```

### 2. Loading the Dataset & Agent Initialization
We initialize our robust LLM model — `llama-3.3-70b-versatile` — passing it zero temperature so it sticks securely strictly to analytical facts. We also enable `allow_dangerous_code=True` uniquely because Pandas agents must execute generated Python code to formulate data answers.

```python
# Bank dataset separated by semicolon
df = pd.read_csv("bank-additional-full.csv", sep=";")

# Initialize Groq LLM  
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Map the data to the LLM agent
eda_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True, 
    agent_type="openai-tools",
    allow_dangerous_code=True 
)
```

### 3. Execution Phase 1: Basic Dataset Overview
Before looking into heavy trends, the agent requests the `y` class balance (our target prediction vector regarding term subscriptions) and identifies null/missing traits.

```python
query_1 = (
    "Analyze this dataset. What is the target variable class balance ('y' column)? "
    "What are the other columns and their data types? Are there any missing/null values?"
)
eda_agent.invoke({"input": query_1})
```

### 4. Execution Phase 2: Distribution & Demographics
Once data integrity is verified, we look at the core predictors: Age, Job Types, Marital Status, and Education metrics against the target variable.

```python
query_2 = (
    "Provide a summary of the 'age', 'job', 'marital', and 'education' columns. "
    "What is the average age, and determining these specific factors, how do they relate to the target 'y'?"
)
eda_agent.invoke({"input": query_2})
```

### 5. Execution Phase 3: Campaign Impact & Correlations
The agent concludes the investigation by tracking previous campaign performances. It answers heavily debated questions like: *Does a long call duration equate to a successful 'yes'? Do users acquired during past campaigns (`poutcome`) convert better?*

```python
query_3 = (
    "Look at the campaign related columns like 'campaign', 'pdays', 'previous', and 'poutcome'. "
    "Do clients with a successful 'poutcome' have a higher chance of a 'yes' in 'y'? "
    "Also, check if duration significantly impacts the target 'y'."
)
eda_agent.invoke({"input": query_3})
```

---

## ▶️ Running the Application

Invoke your conversational EDA agent by simply running:

```bash
python3 eda_agent.py
```

*Note: The agent will output its sequential `chain-of-thought` in verbose mode, allowing you to trace exactly what pandas commands it writes to get you the final answers!*
