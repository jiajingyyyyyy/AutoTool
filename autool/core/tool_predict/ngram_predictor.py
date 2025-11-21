# ngram_predictor.py
from typing import Dict, List, Any, Tuple, Optional, Set
from collections import defaultdict
import time
import pickle
import json
import os
from pathlib import Path
from dotenv import load_dotenv

class NgramPredictor:
    def __init__(self, n=2, debug=False):
        """
        Initialize N-gram predictor.
        :param n: N value for N-gram, default is 2 (bigram)
        :param debug: Enable debug mode
        """
        if n < 2:
            raise ValueError("N-gram n value must be at least 2.")
        self.n = n
        self.debug = debug
        # Store (n-1) tuple -> {successor tool -> frequency} mapping
        self.ngram_counts = defaultdict(lambda: defaultdict(int))
        self.context_counts = defaultdict(int)

    def record_tool_sequence(self, tool_sequence: list):
        """
        Learn N-gram frequencies from a complete tool call sequence.
        
        :param tool_sequence: List of tool name strings
        """
        if len(tool_sequence) < self.n:
            return

        for i in range(len(tool_sequence) - self.n + 1):
            context = tuple(tool_sequence[i : i + self.n - 1])
            next_tool = tool_sequence[i + self.n - 1]
            
            self.ngram_counts[context][next_tool] += 1
            self.context_counts[context] += 1
            
        if self.debug:
            print(f"[NgramPredictor] Learned from sequence {tool_sequence}.")

    def predict_next_tool(self, recent_history_types: list, threshold: float, **kwargs) -> (bool, dict, Optional[str]):
        """
        Predict next tool based on recent history.
        
        :param recent_history_types: Recent tool call history list
        :param threshold: Confidence threshold
        :param kwargs: Additional parameters for compatibility (ignored)
        :return: (high_confidence, overhead_dict, predicted_tool_name)
        """
        start_time = time.time()
        
        if len(recent_history_types) < self.n - 1:
            if self.debug: 
                print("[NgramPredictor] Insufficient history for prediction.")
            return False, {"search_time": time.time() - start_time}, None

        context = tuple(recent_history_types[-(self.n - 1):])

        if context not in self.ngram_counts:
            if self.debug: 
                print(f"[NgramPredictor] Context not found: {context}")
            return False, {"search_time": time.time() - start_time}, None

        successors = self.ngram_counts[context]
        if not successors:
            return False, {"search_time": time.time() - start_time}, None
            
        predicted_type = max(successors, key=successors.get)
        
        # Calculate confidence (conditional probability)
        total_context_count = self.context_counts[context]
        prediction_count = successors[predicted_type]
        confidence_score = prediction_count / total_context_count if total_context_count > 0 else 0.0

        if self.debug:
            print(f"[NgramPredictor] Context: {context}, Candidates: {dict(successors)}")
            print(f"[NgramPredictor] Prediction: '{predicted_type}', Confidence: {confidence_score:.4f} (Threshold: {threshold})")

        overhead = {
            "search_time": time.time() - start_time,
            "SimSCE_time": 0,
            "intuition_embedding_time": 0,
            "path_embedding_time": 0,
            "similarity_cost_time": 0,
        }
        
        has_inertia = confidence_score > threshold
        
        if self.debug:
            status = "successful and above threshold" if has_inertia else "successful but below threshold"
            print(f"[NgramPredictor] Prediction {status}.")
        
        return has_inertia, overhead, predicted_type
        
    
    def save_model(self, file_path: str):
        """Save trained model to file."""
        # Create directory if it doesn't exist
        output_dir = Path(file_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump({
                'n': self.n,
                'ngram_counts': dict(self.ngram_counts),
                'context_counts': dict(self.context_counts)
            }, f)
        print(f"\n[SUCCESS] N-gram model saved to: {file_path}")

               
    def load_from_json(tool_description_path, log_file):
        pass

    @staticmethod
    def load_from_pkl(file_path: str, debug=False):
        """Load trained model from file."""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        predictor = NgramPredictor(n=data['n'], debug=debug)
        predictor.ngram_counts.update(data['ngram_counts'])
        predictor.context_counts.update(data['context_counts'])
        
        print(f"[SUCCESS] N-gram model loaded from {file_path}.")
        return predictor
    

def train_from_log_file(log_file_path: str, output_model_path: str, n_value: int = 3, debug: bool = False):
    """
    Train N-gram model from JSON log file and save it.
    
    :param log_file_path: Path to JSON log file
    :param output_model_path: Path to save trained model
    :param n_value: N value for N-gram
    :param debug: Enable debug mode
    :return: Trained NgramPredictor instance
    """
    print("="*50)
    print(f"Training {n_value}-gram model...")
    print(f"Data source: {log_file_path}")
    print("="*50)

    predictor = NgramPredictor(n=n_value, debug=debug)
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {log_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON format: {log_file_path}")
        return None
        
    total_trajectories_processed = 0
    total_tool_calls = 0

    for trajectory in log_data.get("sequences", []):
        tool_sequence = []
        for step in trajectory.get("steps", []):
            if step.get("type") == "act_ob_pair":
                try:
                    tool_name = step["action"]["parsed_content"]["tool_name"]
                    tool_sequence.append(tool_name)
                except (KeyError, TypeError):
                    continue
        
        if tool_sequence:
            if debug:
                print(f"\nProcessing trajectory ID: {trajectory.get('trajectory_id', 'N/A')}")
            predictor.record_tool_sequence(tool_sequence)
            total_trajectories_processed += 1
            total_tool_calls += len(tool_sequence)

    print("\n" + "="*50)
    print("Training statistics:")
    print(f"  - Total trajectories processed: {total_trajectories_processed}")
    print(f"  - Total tool calls learned: {total_tool_calls}")
    print(f"  - Unique contexts learned: {len(predictor.context_counts)}")
    print("="*50)

    predictor.save_model(output_model_path)
    
    return predictor


def get_config_from_env():
    """
    Load configuration from environment variables with fallback defaults.
    
    :return: Dictionary containing configuration values
    """
    # Load .env file if exists
    load_dotenv()
    
    # Get project root directory (assuming ngram_predictor.py is in src/AutoTool/graph/tool_predict/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[4]  # Adjust based on actual structure
    
    # Default paths relative to project root
    default_model_dir = project_root / "ngram_models"
    default_model_file = default_model_dir / "ngram_model.pkl"
    
    config = {
        'log_file': os.getenv('NGRAM_LOG_FILE', ''),
        'model_output': os.getenv('NGRAM_MODEL_OUTPUT', str(default_model_file)),
        'n_value': int(os.getenv('NGRAM_N_VALUE', '3')),
        'debug': os.getenv('NGRAM_DEBUG', 'false').lower() == 'true'
    }
    
    return config


if __name__ == "__main__":
    # Load configuration from environment variables
    config = get_config_from_env()
    
    LOG_FILE = config['log_file']
    MODEL_OUTPUT_FILE = config['model_output']
    N_VALUE = config['n_value']
    DEBUG_MODE = config['debug']
    
    # Validate log file path
    if not LOG_FILE:
        print("[ERROR] NGRAM_LOG_FILE not set in environment variables.")
        print("Please set it in .env file or provide it as an argument.")
        exit(1)
    
    if not Path(LOG_FILE).exists():
        print(f"[ERROR] Log file not found: {LOG_FILE}")
        exit(1)
    
    print(f"[CONFIG] Log file: {LOG_FILE}")
    print(f"[CONFIG] Model output: {MODEL_OUTPUT_FILE}")
    print(f"[CONFIG] N-gram value: {N_VALUE}")
    print(f"[CONFIG] Debug mode: {DEBUG_MODE}\n")
    
    trained_predictor = train_from_log_file(
        log_file_path=LOG_FILE,
        output_model_path=MODEL_OUTPUT_FILE,
        n_value=N_VALUE,
        debug=DEBUG_MODE
    )

    # Optional: Verify loaded model
    if trained_predictor:
        print("\n--- Verifying loaded model ---")
        loaded_predictor = NgramPredictor.load_from_pkl(MODEL_OUTPUT_FILE)
        
        print("\n--- Sample learned rules (Top 5) ---")
        count = 0
        for context, successors in loaded_predictor.ngram_counts.items():
            if count >= 5:
                break
            most_common_successor = max(successors, key=successors.get)
            print(f"  After sequence {context}, most likely next: '{most_common_successor}'")
            count += 1