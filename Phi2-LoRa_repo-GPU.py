import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer():
    base_model = "microsoft/phi-2"
    peft_model_id = "WillRanger/Phi2-lora-Adapters2"
    config = PeftConfig.from_pretrained(peft_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="cuda:0",return_dict=True, trust_remote_code=True)

    model = model.to('cuda')

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, peft_model_id, trust_remote_code=True)

    model = model.to('cuda')

    return model, tokenizer

def generate(instruction, model, tokenizer):                              
    #input_ids = tokenizer.encode(instruction, return_tensors="pt")


    #input_ids = input_ids.to('cuda')

    #attention_mask = torch.ones_like(input_ids)

    inputs = tokenizer(instruction, return_tensors="pt", return_attention_mask=False)
    outputs = model.generate(**inputs, max_length=512)
    text = tokenizer.batch_decode(outputs)[0]
    
    return text

if __name__ == '__main__':
    model, tokenizer = load_model_and_tokenizer()
    while True:  
        instruction = input("Enter your instruction: ")
        if not instruction:
            continue   
        if instruction.lower() in ["exit", "quit"]:
            print("Exiting...")
            break 

        answer = generate(instruction, model, tokenizer)
        print(answer)
