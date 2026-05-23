from models.qwen import Qwen3_L, Qwen3_S, Qwen2_5_L, Qwen2_5_S

from dotenv import load_dotenv
load_dotenv() # loads HF_TOKEN into environment variables


def run_test():
    
    model_classes = [
        Qwen3_L,
        Qwen3_S,
        Qwen2_5_L,
        Qwen2_5_S
    ]
    
    for model_class in model_classes:
        model = model_class(verbose = True, max_new_tokens = 64)
        
        messages = [{"role": "user", "content": "write a poem about beans"}]
        print(f"{model.name}: {model.get_response(messages)["content"]}")
        
        del model # to prevent CUDA OOM error

if __name__ == "__main__":
    run_test()