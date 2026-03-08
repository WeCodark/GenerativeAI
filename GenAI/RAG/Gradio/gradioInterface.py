import gradio as gr

# 1. Define a normal Python function

def greetUser(name):
    return f'Hello, {name}! Welcome to Generative AI Session'

# Wrap the above function in an interface

# Interface does all the heavy lifting and layout design automatically

demo = gr.ChatInterface(
    fn = greetUser,         # The python function we want to run
    inputs = "textbox",     # This create UI Component for input type
    outputs = "textbox"     # This create UI Component for output type
)


demo.launch(share=True) # It will give us a public link to share our app


