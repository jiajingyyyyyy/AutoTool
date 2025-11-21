# history.py
from typing import List, Dict, Any, Optional


class ExecutionHistory:
    """Maintains execution history of tool calls."""
    
    def __init__(self, max_len: int = 50):
        self.history: List[Dict[str, Any]] = []
        self.max_len = max_len
    
    def add_record(self, tool_name: str, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Add a new execution record."""
        record = {
            "tool_name": tool_name,
            "inputs": inputs,
            "outputs": outputs
        }
        self.history.append(record)
        
        if len(self.history) > self.max_len:
            self.history.pop(0)
    
    def get_output_value(self, tool_name: str, output_param_name: str, lookback: int = 5) -> Optional[Any]:
        """
        Retrieve specific output parameter value from recent execution history.
        
        Args:
            tool_name: Target tool name
            output_param_name: Target output parameter name
            lookback: Maximum number of steps to look back
        
        Returns:
            Parameter value if found, None otherwise
        """
        relevant_history = self.history[-lookback:]
        for record in reversed(relevant_history):
            if record.get("tool_name") == tool_name:
                outputs = record.get("outputs", {})
                if output_param_name in outputs:
                    return outputs[output_param_name]
        return None
    
    def get_recent_tools(self, count: int = 10) -> List[str]:
        """Get list of recently executed tool names."""
        recent = self.history[-count:]
        return [record.get("tool_name") for record in recent if "tool_name" in record]
    
    def clear(self) -> None:
        """Clear all execution history."""
        self.history.clear()
    
    def __len__(self) -> int:
        return len(self.history)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.history[index]