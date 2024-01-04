from transformers import AutoModelForCausalLM, OPTForCausalLM, AutoTokenizer
from peft import PeftConfig, LoraConfig

model_id = "microsoft/phi-2"
peft_model_id = "WillRanger/Phi2-lora-Adapters2"

model = AutoModelForCausalLM.from_pretrained(model_id)
model.load_adapter(peft_model_id)

lora_config = LoraConfig(target_modules=['Wqkv','fc1', 'fc2'])

model.add_adapter(lora_config, adapter_name="lora")
model.enable_adapters()
model.set_adapter("lora")

model.push_to_hub("WillRanger/Phi2-lora",
                  use_auth_token=True,
                  commit_message="basic training",
                  private=False)

print("new model saved") 
