import os
from typing import TypedDict, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables for the LLM
# First try a .env in the current project directory, then fallback to the AgenticAI project .env
load_dotenv()  # tries .env in cwd or parent directories
_alt_env = os.path.join(os.path.expanduser("~"), "PRNV", "AgenticAI", ".env")
if os.path.exists(_alt_env):
    load_dotenv(_alt_env, override=True)

# --- 1. Define the LangGraph State ---
class AgentState(TypedDict):
    data_path: str
    eda_summary: str
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    model_accuracies: Dict[str, float]
    best_model_name: str
    final_report: str

# --- 2. Define the Nodes ---

def perform_eda(state: AgentState):
    """ Node 1: Perform Exploratory Data Analysis """
    print("--- [NODE] Performing EDA ---")
    data_path = state["data_path"]
    # The dataset uses a semicolon delimiter based on our preview
    df = pd.read_csv(data_path, sep=';')
    
    # Gather basic stats
    shape_info = f"Dataset Shape: {df.shape}"
    target_dist = f"Target variable 'y' distribution:\n{df['y'].value_counts(normalize=True)*100}"
    missing_vals = f"Missing values sum:\n{df.isnull().sum().sum()}"
    
    eda_summary = f"{shape_info}\n\n{target_dist}\n\n{missing_vals}"
    print(eda_summary)
    
    # Store strictly necessary EDA info back into state
    return {"eda_summary": eda_summary}

def preprocess_data(state: AgentState):
    """ Node 2: Data Preprocessing """
    print("--- [NODE] Preprocessing Data ---")
    data_path = state["data_path"]
    df = pd.read_csv(data_path, sep=';')
    
    # Drop rows with NA if any
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']
    
    # Encode Target Variable
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    # Encode Categorical Features
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Using one-hot encoding for features (standard practice for these types of algorithms)
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Data preprocessed successfully. Training shape: {X_train.shape}")
    
    # Pass DataFrames/arrays directly in state (in a production setting, paths might be preferred)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

def build_models(state: AgentState):
    """ Node 3: Build 3 Models and evaluate Accuracy """
    print("--- [NODE] Building Models ---")
    X_train = state["X_train"]
    X_test = state["X_test"]
    y_train = state["y_train"]
    y_test = state["y_test"]
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42) # reduced trees for speed
    }
    
    accuracies = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        accuracies[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")
        
    return {"model_accuracies": accuracies}

def evaluate_and_report(state: AgentState):
    """ Node 4: Identify best model and generate report via LLM """
    print("--- [NODE] Evaluating & Reporting ---")
    accuracies = state["model_accuracies"]
    eda_summary = state["eda_summary"]
    
    best_model_name = max(accuracies, key=accuracies.get)
    best_accuracy = accuracies[best_model_name]
    
    print(f"\n[EVALUATION] Best Model is {best_model_name} with Accuracy: {best_accuracy:.4f}\n")
    
    # We will use the conversational LLM to format this into a professional summary snippet
    try:
        llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
        
        prompt = f"""
        You are a Data Science Assistant. Formulate a short executive summary based on the following pipeline results:
        
        EDA Highlights:
        {eda_summary}
        
        Model Evaluation:
        {accuracies}
        
        Best Model:
        {best_model_name} ({best_accuracy:.4f})
        
        Write a concise, bulleted professional summary indicating which model won and why the accuracy metrics matter.
        """
        
        response = llm.invoke(prompt)
        final_report = response.content
        print("\n--- FINAL AGENT REPORT ---")
        print(final_report)
    except Exception as e:
        print("Could not generate LLM report (likely missing API key). Proceeding with fallback text report.")
        final_report = f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}. All accuracies: {accuracies}"
        print("\n--- FINAL AGENT REPORT ---")
        print(final_report)

    return {"best_model_name": best_model_name, "final_report": final_report}

# --- 3. Construct Graph ---
workflow = StateGraph(AgentState)

# Add node definitions
workflow.add_node("perform_eda", perform_eda)
workflow.add_node("preprocess_data", preprocess_data)
workflow.add_node("build_models", build_models)
workflow.add_node("evaluate_and_report", evaluate_and_report)

# Add edges defining flow
workflow.add_edge(START, "perform_eda")
workflow.add_edge("perform_eda", "preprocess_data")
workflow.add_edge("preprocess_data", "build_models")
workflow.add_edge("build_models", "evaluate_and_report")
workflow.add_edge("evaluate_and_report", END)

# Compile the final agent graph
app = workflow.compile()

if __name__ == "__main__":
    # Start graph execution
    initial_state = {"data_path": "bank-additional-full.csv"}
    
    print("Initiating LangGraph Data Science Agent Workflow...\n")
    app.invoke(initial_state)
