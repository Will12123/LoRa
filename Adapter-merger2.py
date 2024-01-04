from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, LoraConfig

# Load the base model
model_id = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(model_id)

# Load and configure the adapter
peft_model_id = "WillRanger/Phi2-lora-Adapters2"
model.load_adapter(peft_model_id)

lora_config = LoraConfig(target_modules=['Wqkv','fc1', 'fc2'])
model.add_adapter(lora_config, adapter_name="lora")
model.enable_adapters()
model.set_adapter("lora")


# Save the entire model state (including the base model and the configured adapter)
model.save_pretrained('./model')

# Push the entire model (with the adapter configuration) to the hub
model = AutoModelForCausalLM.from_pretrained("./model")
model.push_to_hub("WillRanger/Phi2-lora",
                  use_auth_token=True,
                  commit_message="Basic training with integrated adapter",
                  private=False)

print("New model with integrated adapter saved and pushed to the hub")
