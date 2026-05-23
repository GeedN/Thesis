from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

class LLM:
    def __init__(self, model_name, model_id, verbose = False, **kwargs):
        self.enable_thinking = kwargs.get("enable_thinking", False)
        self.max_new_tokens = kwargs.get("max_new_tokens", 4)
        self.name = model_name
        if verbose:
            print(f"loading model: {self.name}")
            
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="cuda",
        )
        
        if verbose:
            print(f"model loaded: {self.name}")
    
    def get_response(self, messages):
    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            do_sample=False, # deterministic outputs
            max_new_tokens=self.max_new_tokens
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        try:
            # 151668 = </think>
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        return {
            "thinking_content": thinking_content,
            "content": content
        }