from autool.message import Message

class TemporaryMemory:
   
    def __init__(
        self
    ):    
        self.memory=[]

    def add_memory(
        self,
        msg:Message
    ):
        self.memory.append(msg)

    def clear_memory(
        self
    ):
        self.memory=[]
    
    def get_memory(
        self,
        parse:bool=True
    ):
        if parse:
            parse_res = []
            for i in range(len(self.memory)):
                parse_res.append(self.memory[i].parse())
            return parse_res
        
        else:
            return self.memory

class EssentialMemory:
   
    def __init__(
        self
    ):    
        self.memory=[]
        self.progress = []

    def add_memory(
        self,
        msg:Message
    ):
        self.memory.append(msg)

    def add_progress(
        self,
        msg:Message
    ):
        self.progress.append(msg)

    def get_progress(
        self
    ):
        return self.progress

    def clear_memory(
        self
    ):
        self.memory=[]

    def get_memory(
        self,
        parse:bool=True
    ):
        if parse:
            parse_res = []
            for i in range(len(self.memory)):
                parse_res.append(self.memory[i].parse())
            return parse_res
        
        else:
            return self.memory
