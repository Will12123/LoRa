
#import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import json
from huggingface_hub import notebook_login
import torch
import torch.nn as nn

notebook_login()

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/phi-2-GPTQ", 
    device_map='auto',
    trust_remote_code=True,
)


tokenizer = AutoTokenizer.from_pretrained("TheBloke/phi-2-GPTQ", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

#model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model 

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    target_modules=['Wqkv','out_proj'], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

import transformers
from datasets import load_dataset
data = load_dataset("WillRanger/Electrical-engineering")

#with open(data1, 'r') as file:
#    json_data = json.load(file)

#data = {'train': json_data}

def merge_columns(example):
    new_example = {
        "instruction": example['instruction'],
        "input": example['input'],
        "prediction": example['instruction'] + " | " + example['input'] + " ->: " + example['output']
    }
    
    return new_example


data['train'] = data['train'].map(merge_columns)
#data['train'] = list(map(merge_columns, data['train']))
data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)
#data

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data['train'],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=True,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

model.push_to_hub("WillRanger/Phi2-lora-Adapters",
                  use_auth_token=True,
                  commit_message="basic training",
                  private=True)