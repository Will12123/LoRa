from transformers import AutoModelForCausalLM
#from peft import LoraConfig, get_peft_model, merge_adapter, merge_and_unload, PeftModel
from peft import PeftModel
import torch

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, use_safetensors=True)

#lora_config = LoraConfig(r=32, alpha=64, target_modules=['Wqkv','fc1', 'fc2'])

model2 = PeftModel.from_pretrained(model, "WillRanger/Phi2-lora-Adapters2", trust_remote_code=True, use_safetensors=True)
#peft_model = get_peft_model(model, lora_config)


#merge_adapter(peft_model, adapter_path="WillRanger/Phi2-lora-Adapters2")
merged_model = model2.merge_and_unload()
merged_model.save_pretrained('./model')

#model2.push_to_hub("WillRanger/Phi2-lora",
 #                 use_auth_token=True,
  ##                commit_message="basic training",
    #              private=False)

print("new model saved")        
