def check_tool_failure(last_memory, task_type):
    if task_type == "academic" and "Invalid action" in last_memory or \
                task_type == "tool-query" and "Error" in last_memory or \
                task_type == "alfworld" and "Nothing happens" in last_memory or \
                task_type == "scienceworld" and "No known action" in last_memory or \
                task_type == "scienceworld" and "Unknown action" in last_memory or \
                task_type == "scienceworld" and "It's not clear " in last_memory:
        return True
    return False
        