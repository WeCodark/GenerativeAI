import os
import pandas as pd
from dotenv import load_dotenv

# LangChain imports
# Note: For pandas dataframe agent, we use langchain_experimental
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# You can swap this with ChatOpenAI, ChatGoogleGenerativeAI, etc., based on your preferred provider.
from langchain_groq import ChatGroq

# Load environment variables (e.g., from a .env file)
load_dotenv()

def run_eda_agent(csv_file_path: str):
    """
    Initializes a LangChain Pandas DataFrame agent to perform Exploratory Data Analysis (EDA).
    """
    print(f"Loading data from {csv_file_path}...")
    try:
        # Bank dataset uses semicolon as separator
        df = pd.read_csv(csv_file_path, sep=';')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print(f"Data loaded successfully. Shape: {df.shape}")
    print("Initializing the LangChain EDA agent...")

    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

    # Create the Agent
    # We set allow_dangerous_code=True which is required by newer versions of langchain_experimental 
    # to evaluate pandas code safely in your environment.
    eda_agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=True, # Set to True to see the agent's thought process
        agent_type="openai-tools", # Use zero-shot-react-description if not using an OpenAI tools capable model
        allow_dangerous_code=True 
    )

    print("\n" + "="*50)
    print("🤖 Starting Autonomous EDA 🤖")
    print("="*50 + "\n")

    # Step 1: Basic Dataset Overview
    print("\n--- Phase 1: Target Variable & Basic Information ---")
    query_1 = (
        "Analyze this dataset. What is the target variable class balance ('y' column)? "
        "What are the other columns and their data types? Are there any missing/null values?"
    )
    eda_agent.invoke({"input": query_1})

    # Step 2: Distribution & Demographics
    print("\n--- Phase 2: Client Demographics & Status ---")
    query_2 = (
        "Provide a summary of the 'age', 'job', 'marital', and 'education' columns. "
        "What is the average age, and determining these specific factors, how do they relate to the target 'y'?"
    )
    eda_agent.invoke({"input": query_2})

    # Step 3: Correlation & Campaign Information
    print("\n--- Phase 3: Campaign Impact & Correlations ---")
    query_3 = (
        "Look at the campaign related columns like 'campaign', 'pdays', 'previous', and 'poutcome'. "
        "Do clients with a successful 'poutcome' have a higher chance of a 'yes' in 'y'? "
        "Also, check if duration significantly impacts the target 'y'."
    )
    eda_agent.invoke({"input": query_3})

    print("\n" + "="*50)
    print("✅ EDA Complete ✅")
    print("="*50)


if __name__ == "__main__":
    bank_dataset = "bank-additional-full.csv"
    
    if os.path.exists(bank_dataset):
        print(f"Found dataset: {bank_dataset}")
        run_eda_agent(bank_dataset)
    else:
        print(f"Error: {bank_dataset} not found in the current directory.")
