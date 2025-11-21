import re
from typing import Dict, Any, Optional

def _get_objects_from_text(text: str):
         """从文本中提取物体名称 (例如 "apple 1")"""
         if not text: return set()
         # 改进正则以匹配 "X Y" (字母+数字) 或只有字母的模式
         # obj_pattern = r"\b([a-zA-Z]+(?:\s+[a-zA-Z]+)*)\s+(\d+)\b" # 旧的，只匹配带数字的
         # 更通用的模式，匹配 "a/an/the X 1" 或 "a/an/the X"
         obj_pattern = r"\b(?:a|an|the)\s+([a-zA-Z]+(?:(?:\s+[0-9]+)?))\b"
         found = re.findall(obj_pattern, text.lower())
         return set(found)

def parse_alfworld_action(action_text: Optional[str], observation: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Parses an Alfworld natural language action into a structured dictionary.
    Uses keyword matching and optional validation against observation.
    
    Args:
        action_text: The action string to parse
        observation: Optional observation text for validation
    
    Returns:
        Parsed action dictionary with tool_name and inputs
    """
    if not action_text:
        return None

    action_text_lower = action_text.lower().strip()
    parsed_action = {"tool_name": "unknown", "inputs": {}}

    # Define tool keywords and their parsing logic
    tool_parsers = {
        "go": parse_go_action,
        "take": parse_take_action,
        "put": parse_put_action,
        "open": parse_open_action,
        "close": parse_close_action,
        "toggle": parse_toggle_action,
        "clean": parse_clean_action,
        "cool": parse_cool_action,
        "heat": parse_heat_action,
        "examine": parse_examine_action,
        "inventory": parse_inventory_action,
        "look": parse_look_action,
        "check": parse_check_action,
        "use": parse_use_action,
    }

    # Try to match keywords
    for keyword, parser in tool_parsers.items():
        if keyword in action_text_lower:
            result = parser(action_text_lower)
            if result:
                parsed_action = result
                break

    # Validate against observation if provided
    if observation and parsed_action["tool_name"] != "unknown":
        if not check_action_valid(parsed_action, observation):
            print(f"Warning: Action '{action_text}' may be invalid based on observation")

    if parsed_action["tool_name"] == "unknown":
        print(f"Warning: Could not parse action: '{action_text}'")

    return parsed_action


def parse_go_action(action_text: str) -> Optional[Dict[str, Any]]:
    """
    Parse 'go' action with flexible format:
    - "go to <recep>" -> single parameter: {"recep": ...}
    - "go <through> to <recep>" -> dual parameters: {"recep": ..., "through": ...}
    
    Examples:
        "go to desk 1" -> {"recep": "desk 1"}
        "go door to living room" -> {"recep": "living room", "through": "door"}
    """
    if "go" not in action_text:
        return None
    
    # Remove "go" keyword and clean up
    text = action_text.replace("go", "", 1).strip()
    
    # Check for "to" separator
    if "to" in text:
        parts = text.split("to", 1)
        
        if len(parts) == 2:
            before_to = parts[0].strip()
            after_to = parts[1].strip()
            
            # If there's text between "go" and "to", it's the "through" parameter
            if before_to:
                return {
                    "tool_name": "go",
                    "inputs": {
                        "recep": after_to,
                        "through": before_to
                    }
                }
            
            # No text between "go" and "to" -> single parameter
            return {
                "tool_name": "go",
                "inputs": {"recep": after_to}
            }
    
    # Fallback: treat everything after "go" as recep
    if text:
        return {
            "tool_name": "go",
            "inputs": {"recep": text}
        }
    
    return None


def parse_take_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'take <obj> from <recep>'"""
    if "take" not in action_text:
        return None
    
    match = re.match(r".*take\s+(.+?)\s+from\s+(.+)", action_text)
    if match:
        return {
            "tool_name": "take",
            "inputs": {
                "obj": match.group(1).strip(),
                "recep": match.group(2).strip()
            }
        }
    
    # Fallback: just take object
    parts = action_text.replace("take", "").strip().split()
    if parts:
        return {
            "tool_name": "take",
            "inputs": {"obj": " ".join(parts), "recep": ""}
        }
    
    return None


def parse_put_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'put <obj> in/on <recep>'"""
    if "put" not in action_text:
        return None
    
    # Try standard patterns with in/on
    match = re.match(r".*put\s+(.+?)\s+(?:in|on|in/on)\s+(.+)", action_text)
    if match:
        return {
            "tool_name": "put",
            "inputs": {
                "obj": match.group(1).strip(),
                "recep": match.group(2).strip()
            }
        }
    
    # Fallback: split by common prepositions
    text = action_text.replace("put", "").strip()
    for prep in ["in", "on", "into", "onto"]:
        if f" {prep} " in text:
            parts = text.split(f" {prep} ", 1)
            if len(parts) == 2:
                return {
                    "tool_name": "put",
                    "inputs": {
                        "obj": parts[0].strip(),
                        "recep": parts[1].strip()
                    }
                }
    
    return None


def parse_open_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'open <recep>'"""
    if "open" not in action_text:
        return None
    
    target = action_text.replace("open", "").strip()
    if target:
        return {"tool_name": "open", "inputs": {"recep": target}}
    return None


def parse_close_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'close <recep>'"""
    if "close" not in action_text:
        return None
    
    target = action_text.replace("close", "").strip()
    if target:
        return {"tool_name": "close", "inputs": {"recep": target}}
    return None


def parse_toggle_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'toggle <target>'"""
    if "toggle" not in action_text:
        return None
    
    target = action_text.replace("toggle", "").strip()
    if target:
        return {"tool_name": "toggle", "inputs": {"target": target}}
    return None


def parse_clean_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'clean <obj> with <recep>'"""
    if "clean" not in action_text:
        return None
    
    match = re.match(r".*clean\s+(.+?)\s+with\s+(.+)", action_text)
    if match:
        return {
            "tool_name": "clean",
            "inputs": {
                "obj": match.group(1).strip(),
                "recep": match.group(2).strip()
            }
        }
    return None


def parse_cool_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'cool <obj> with <recep>'"""
    if "cool" not in action_text:
        return None
    
    match = re.match(r".*cool\s+(.+?)\s+with\s+(.+)", action_text)
    if match:
        return {
            "tool_name": "cool",
            "inputs": {
                "obj": match.group(1).strip(),
                "recep": match.group(2).strip()
            }
        }
    return None


def parse_heat_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'heat <obj> with <recep>'"""
    if "heat" not in action_text:
        return None
    
    match = re.match(r".*heat\s+(.+?)\s+with\s+(.+)", action_text)
    if match:
        return {
            "tool_name": "heat",
            "inputs": {
                "obj": match.group(1).strip(),
                "recep": match.group(2).strip()
            }
        }
    return None


def parse_examine_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'examine <target>'"""
    if "examine" not in action_text:
        return None
    
    target = action_text.replace("examine", "").strip()
    if target:
        return {"tool_name": "examine", "inputs": {"target": target}}
    return None


def parse_inventory_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'inventory'"""
    if "inventory" in action_text:
        return {"tool_name": "inventory", "inputs": {}}
    return None


def parse_look_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'look'"""
    if action_text.strip() == "look":
        return {"tool_name": "look", "inputs": {}}
    return None


def parse_check_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'check valid actions' or 'check'"""
    if "check" in action_text:
        return {"tool_name": "check", "inputs": {}}
    return None


def parse_use_action(action_text: str) -> Optional[Dict[str, Any]]:
    """Parse 'use <target>'"""
    if "use" not in action_text:
        return None
    
    target = action_text.replace("use", "").strip()
    if target:
        return {"tool_name": "use", "inputs": {"target": target}}
    return None


def check_action_valid(parsed_action: Dict[str, Any], observation: str) -> bool:
    """
    Validate parsed action against observation.
    Check if the action parameters exist in the observation.
    """
    if not observation:
        return True
    
    obs_lower = observation.lower()
    inputs = parsed_action.get("inputs", {})
    
    # Check if key parameters appear in observation
    for param_value in inputs.values():
        if param_value and isinstance(param_value, str):
            if param_value.lower() not in obs_lower:
                return False
    
    return True

def infer_output_from_observation(action_parsed: Dict[str, Any], observation_text: str) -> Dict[str, Any]:
    """
    根据动作和观察推断结构化的输出。
    这是构建参数依赖图的关键。需要根据不同动作类型和观察文本模式实现。
    """
    tool_name = action_parsed.get("tool_name")
    obs_lower = observation_text.lower()
    output = {
        "status": "unknown"
    }
    
    # 分析观察文本推断状态
    if "nothing happens" in obs_lower:
        output["status"] = "failure"
    elif "you open" in obs_lower and "and see" in obs_lower:
        output["status"] = "success"
        output["receptacle_state"] = "open"
        # 可以尝试提取看到的物品
        match = re.search(r"you open the (.*?) and see (.*?)\.", obs_lower)
        if match:
            output["opened_receptacle"] = match.group(1).strip()
            output["observed_content"] = _get_objects_from_text(match.group(2))
    elif "you close" in obs_lower:
        output["status"] = "success"
        output["receptacle_state"] = "closed"
        match = re.search(r"you close the (.*?)\.", obs_lower)
        if match: 
            output["closed_receptacle"] = match.group(1).strip()
    elif tool_name == "take" and "you pick up the" in obs_lower:
        output["status"] = "success"
        obj = action_parsed.get("inputs", {}).get("obj")
        if obj: 
            output["taken_object"] = obj
    elif tool_name == "put" and "you put the" in obs_lower:
        output["status"] = "success"
        # put 动作本身不直接产生新信息输出，主要是状态改变
    elif tool_name == "go" and ("you arrive at" in obs_lower or obs_lower.startswith("on the")):
        output["status"] = "success"
        loc = action_parsed.get("inputs", {}).get("recep")
        if loc: 
            output["current_location"] = loc # 使用输入作为输出位置
    elif tool_name == "examine":
        output["status"] = "success"
        # 可以提取描述内容
        output["examined_content"] = _get_objects_from_text(observation_text)
    else:
        # 默认成功，如果观察不像失败信息
        if "fail" not in obs_lower and "can't" not in obs_lower:
            output["status"] = "success"
    
        
    return output
# Example Usage (assuming tool_descriptions is loaded from your JSON)
# tool_descriptions = json.load(open("tool_description.json"))
# parsed = parse_alfworld_action("go to desk 1", tool_descriptions)
# print(parsed)
# parsed = parse_alfworld_action("take apple 1 from fridge 1", tool_descriptions)
# print(parsed)
# parsed = parse_alfworld_action("check valid actions", tool_descriptions)
# print(parsed)
# parsed = parse_alfworld_action("look under table 1", tool_descriptions) # Example of unknown
# print(parsed)