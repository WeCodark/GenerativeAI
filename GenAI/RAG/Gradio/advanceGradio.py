import gradio as gr

# This function is used to process all our complex inputs
def createProfile(name, age, role, avatarImage):
    message = f'Success! {name} is a {age} year old {role}.'

    # we will now return the text message, and we just pass the image back to display it
    return message, avatarImage

# Now, we will build UI using Rows and Columns
with gr.Blocks() as demo:
    gr.Markdown('# User Profile Creator')

    with gr.Row():
        # left Column: Inputs
        with gr.Column():
            name_input = gr.Textbox(label='Full Name', placeholder='Your Name')
            # Slider: Here we can select the value between the min max range
            age_input = gr.Slider(minimum=18, maximum=80, step = 1, label='Age',value = 25)

            # dropdown: It provide us a list of Chocies
            role_input = gr.Dropdown(
                choices = ["Developer","Designer",'CyberTech','Data Scientist','Manager','Prompt Engineer'],
                label="Job Role"
            )

            # Image: type= 'Filepath' it sends the image path to your python funtion
            avtarInput = gr.Image(label='Upload Profile Pic', type='filepath')

            # A button that generates the profile
            submitBtn = gr.Button('Generate Profile')
        # My Right column where we will get Outputs
        with gr.Column():

            textOutput = gr.Textbox(label = 'System Message')
            imageOutput = gr.Image(label= 'Profile Picture Display')
    # Now we will connect UI to the python function
    submitBtn.click(
        fn = createProfile,
        inputs=[name_input,age_input,role_input,avtarInput],
        outputs=[textOutput,imageOutput]
    )


demo.launch()
