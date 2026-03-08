import gradio as gr

def calculateSquare(number):
    return number * number

# If we want to build a custom layout we use gr.Blocks

with gr.Blocks() as demo:
    gr.Markdown("##Simple Square Calculator")

    with gr.Row(): # It puts components side by side
        num_input = gr.Number(label= 'Enter a Number')
        resultOutput = gr.Number(label='Result')

    # we will create now a submit button
    submitBtn = gr.Button("Calculate Square")

    # What happens whe we click the button 
    # It takes num input, run it through calculateSquare, and puts the anwer in result output variable

    submitBtn.click(
        fn = calculateSquare,
        inputs = num_input,
        outputs = resultOutput
    )

demo.launch()
