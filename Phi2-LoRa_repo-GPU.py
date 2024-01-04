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

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
return tokenizer.decode(output[0])

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
