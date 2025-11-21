class Toolkit:
    def __init__(
        self,
        tools:dict
    ):
        self.tools = tools
        
        
    def add_tool(
        self,
        tools:dict
    ):  
        raise("Not imple add_tool (todo...")

        
    def call_tool(
        self,
        tool_name: str,
        kwargs: dict
    ):  
        return self.tools[tool_name]["tool"](**kwargs)


def get_tool_desc(toolkit:Toolkit):  
    tool_desc = """
There are some functions you can use,the following tool functions are available in the format of
```
{function index}. {function name}: {function description}
{argument1 name} ({argument type}): {argument description}
{argument2 name} ({argument type}): {argument description}
...
```
0. finish: If you think you finish the task,please call this function.

""" 
    function_index=1
    for key,value in toolkit.tools.items():
        tool_desc = tool_desc + str(function_index) + ". " + key +": " + value["tool_desc"] + ".\n"
        for i in range(len(value["args"])):
            arg_name=value["args"][i]["arg_name"]
            arg_type=value["args"][i]["arg_type"]
            arg_desc=value["args"][i]["arg_desc"]
            tool_desc = tool_desc + arg_name + "(" + arg_type + "): " + arg_desc + "\n"
        
        tool_desc = tool_desc + "\n"
        function_index = function_index+1
    
    return tool_desc