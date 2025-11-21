from enum import Enum

 
class Role(Enum):
    SYSTEM = "system" 
    USER = "user" 
    ASSISTANT = "assistant" 

 

class Message:
    def __init__(
        self,
        role:Enum,
        content:str
    ):
        self.content=content
        self.type=type
        self.role = role.value if isinstance(role, Enum) else role
    
    def parse(
        self
    ):
        parse_res = {
          "role":self.role,
          "content":self.content
        }
        
        return parse_res
    
  