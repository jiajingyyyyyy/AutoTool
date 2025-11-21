import json

def parse_tool_query(tool_query: str) -> dict:
    """
    Parses the tool query string into a dictionary.

    Args:
        tool_query (str): The tool query string to parse.

    Returns:
        dict: A dictionary containing the parsed tool query.
    """
    try:
        if " with Action Input: " in tool_query:
            tool_name, json_part = tool_query.split(" with Action Input: ", 1)
            inputs = json.loads(json_part)
            if tool_name == "finish": inputs = {}
            return {
                "tool_name": tool_name.strip(),
                "inputs": inputs
            }
        
    except Exception as e:
        print(f"Error parsing tool query: {e}")
        return {
            "tool_name": "unknown",
            "inputs": {}
        }
        

if __name__ == '__main__':
    # Example usage
    tool_query = 'neighbourCheck with Action Input: {\"graph\": \"AuthorNet\", \"node\": \"Fadi Boutros\"}'
    parsed_query = parse_tool_query(tool_query)
    print(parsed_query)