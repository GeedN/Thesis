from enum import Enum

class Judgement(Enum):
    FALSE = 0
    TRUE = 1
    UNCERTAIN = 2
    NONE = 3
    
def build_prompt(statement_framing, statement, author):
    
    result = statement_framing.replace("[STATEMENT]", statement).replace("[AUTHOR]", author)
    
    return result

def get_truth_judgement(model, statement_framing, statement, author, binary = True):
    
    prompt = build_prompt(statement_framing, statement, author)
    
    messages = []
    messages.append({"role": "user", "content": prompt})
    
    response = model.get_response(messages)
    
    response = response["content"].lower()
    
    if "true" in response and "false" in response:
        print("MALFORMED RESPONSE:", response)
        return Judgement.NONE

    elif "false" in response:
        return Judgement.FALSE
    
    elif "true" in response:
        return Judgement.TRUE
    
    elif not binary and "uncertain" in response:
        return Judgement.UNCERTAIN
    
    else:
        print("MALFORMED RESPONSE:", response)
        return Judgement.NONE