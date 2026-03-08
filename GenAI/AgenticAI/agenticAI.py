# we import libraries
from langgraph.graph import StateGraph, END
from typing import TypedDict

# Define the state graph
class State(TypedDict):
    text : str


# Define the Nodes
# node 1: Convert the text data into uppercase
def to_uppercase(state):
    text = state['text']
    return {'text': text.upper()}

# Node 2: Add exclamation mark
def add_exclamation(state):
    text = state['text']
    return {'text': text + '!'}

# THis creates a graph workflow object
workflow = StateGraph(State)

workflow.add_node('uppercase', to_uppercase)
workflow.add_node('exclamation', add_exclamation)

# We will connect the nodes
workflow.set_entry_point('uppercase')
workflow.add_edge('uppercase', 'exclamation')
workflow.add_edge('exclamation', END)


# Now my graph is ready to run
app = workflow.compile()

result = app.invoke({'text': 'Hello World'})
print(result)


# After THis Do research about Conditional Routing and Loops