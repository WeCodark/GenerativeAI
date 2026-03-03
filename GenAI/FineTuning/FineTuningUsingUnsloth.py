# Import Libraries for Finetuning
from unsloth import FastLanguageModel, is_bf16_supported # for my t4 gpu 
from datasets import load_dataset
from trl import SFTTrainer # transformer Reinforcement Learning
from transformers import TrainingArguments # hyperparameters for training
import torch

max_seq_length = 512  # 1024 for 8b model

model ,tokenizer = FastLanguageModel.from_pretrained(model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit",
                                    max_seq_length=max_seq_length,
                                    dtype=None,
                                    load_in_4bit=True)

# LoRA (Low Rank Adaptation)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank of the low-rank matrices
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16, # Scaling factor
    lora_dropout = 0, # We are not using dropout to get more training speed
    bias = "none", # Bias type
    use_gradient_checkpointing = "unsloth", # it is unsloth custom memory saver
)

# We will prepare our Dataset (with manual stopping token)

def format_prompts(examples):
    prompt_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
                        {problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
                        {answer}<|eot_id|>"""
    inputs = [msg[0]['content'] for msg in examples['messages']]
    outputs = examples['reannotated_assistant_content']

    texts = []
    for problem, answer in zip(inputs, outputs):
        text = prompt_template.format(problem=problem, answer=answer)
        texts.append(text)
    
    # we will use tokenizer to truncate the texts to max_seq_length
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_length,
        add_special_tokens=False # Our template already has special tokens
    )

    # We will return the actual tokeinzed IDs instead of texts
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

# We will load dataset
dataset = load_dataset("ServiceNow-AI/R1-Distill-SFT", "v1" ,split="train[:5000]")

# We will apply the format_prompts function to the dataset
dataset = dataset.map(format_prompts, batched=True, remove_columns=dataset.column_names)

# We will define Training Arguments
# trainer = SFTTrainer(
#     model = model,
#     tokenizer = tokenizer,
#     train_dataset = dataset,
#     dataset_text_field = "text",
#     max_seq_length = max_seq_length,
#     args = TrainingArguments(
#         per_device_train_batch_size=1,
#         gradient_accumulation_steps=8,
#         warmup_steps=5,
#         max_steps=60, # if you want better loss optimization the use more max steps
#         learning_rate=2e-4,
#         fp16=not is_bf16_supported(),
#         bf16=is_bf16_supported(),
#         logging_steps=1,
#         optim="adamw_8bit", # it is unsloth custom memory saver
#         seed=3407,
#         output_dir="lora_outputs",
#     )    
# )

# We will define Training Arguments withour SFT

trainer = FastLanguageModel.get_train_loop(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=60, # if you want better loss optimization the use more max steps
        learning_rate=2e-4,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit", # it is unsloth custom memory saver
        seed=3407,
        output_dir="lora_outputs",
    )    
)


trainer.train()

# We will save the model
model.save_pretrained("my_llama3_Model")
tokenizer.save_pretrained("my_llama3_Model_tokenizer")

# return_tensor = 'pt'

# Testing 
# We will define our prompt template for testing
prompt_style = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>
                        {problem}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

# question = "if i have 3 shirt and it takes 3 hours to dry them outside, how long will it take to dry 30 shirts?"
question = "harsh has 3 sisters, each of his sisters has 2 brothers, how many brothers does harsh have?"

# We will format the question
prompt = prompt_style.format(problem=question)

# We will tokenize the question
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# We will generate the answer
outputs = model.generate(**inputs, max_new_tokens=50)

# We will decode the answer
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Question:", question)
print("Answer:", answer)