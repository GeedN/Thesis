from models.template_class import LLM

class Qwen2_5_S(LLM):
    def __init__(self, **kwargs):
        
        super().__init__(model_name = "qwen2.5_1.5B", model_id = "Qwen/Qwen2.5-1.5B-Instruct", **kwargs)
        
class Qwen3_S(LLM):
    def __init__(self, **kwargs):
        
        super().__init__(model_name = "qwen3_0.6B", model_id = "Qwen/Qwen3-0.6B", **kwargs)
        
class Qwen2_5_L(LLM):
    def __init__(self, **kwargs):
        
        super().__init__(model_name = "qwen2.5_7B", model_id = "Qwen/Qwen2.5-7B-Instruct", **kwargs)
        
class Qwen3_L(LLM):
    def __init__(self, **kwargs):
        
        super().__init__(model_name = "qwen3_8B", model_id = "Qwen/Qwen3-8B", **kwargs)