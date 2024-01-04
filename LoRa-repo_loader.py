import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer



def load_model_and_tokenizer():
    base_model = "microsoft/phi-2"
    peft_model_id = "WillRanger/Phi2-lora-Adapters2"
    config = PeftConfig.from_pretrained(peft_model_id, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(base_model, return_dict=True, trust_remote_code = True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = PeftModel.from_pretrained(model, peft_model_id, trust_remote_code = True)
    return model, tokenizer


def generate(instruction, model, tokenizer):                              
    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    response = model.generate(input_ids, 
                              attention_mask=attention_mask, 
                              max_length=250,
                              pad_token_id=tokenizer.eos_token_id,
                              num_beams=5,  
                              no_repeat_ngram_size=2,
                              temperature=0.7,
                              top_k=50)
    answer = tokenizer.decode(response[0], skip_special_tokens=True)
    return answer

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