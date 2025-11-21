"""
AutoTool Project Configuration Module
Manages all configuration items, supports loading from environment variables and .env files
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Get package root directory (where config.py is located)
PACKAGE_ROOT = Path(__file__).parent

# Load .env file (prioritize package root directory)
env_file = PACKAGE_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✅ Configuration loaded from: {env_file}")
else:
    # Try loading from current working directory
    load_dotenv()
    print("⚠️  autool/.env not found, attempting to load from current directory")


class Config:
    """Project configuration class, manages all configuration items"""
    
    # ============================================================================
    # Path Configuration (relative to package root)
    # ============================================================================
    PACKAGE_ROOT = PACKAGE_ROOT
    
    # ============================================================================
    # API Configuration
    # ============================================================================
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # ============================================================================
    # Model Path Configuration
    # ============================================================================
    SIMCSE_MODEL_PATH = os.getenv(
        "SIMCSE_MODEL_PATH",
        "princeton-nlp/sup-simcse-roberta-base"
    )
    TRANSFORMERS_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE_DIR")
    
    # ============================================================================
    # Data File Paths
    # ============================================================================
    # Supports both relative paths (relative to package root) and absolute paths
    TOOL_DESC_FILE = os.getenv(
        "TOOL_DESC_FILE",
        str(PACKAGE_ROOT / "data" / "tool_description.json")
    )
    
    TRAJECTORY_DIR = os.getenv(
        "TRAJECTORY_DIR",
        str(PACKAGE_ROOT / "data" / "trajectories")
    )
    
    # ============================================================================
    # Output Paths
    # ============================================================================
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(PACKAGE_ROOT / "results"))
    PIE_CHART_DIR = os.getenv("PIE_CHART_DIR", str(PACKAGE_ROOT / "results" / "pie_charts"))
    MARKOV_ANALYSIS_DIR = os.getenv("MARKOV_ANALYSIS_DIR", str(PACKAGE_ROOT / "results" / "markov"))
    
    # ============================================================================
    # Model Inference Parameters
    # ============================================================================
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen/qwen2.5-7b-instruct")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
    TOP_P = float(os.getenv("TOP_P", "0.95"))
    FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", "0"))
    PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0"))
    SEED = int(os.getenv("SEED", "42"))
    
    # ============================================================================
    # Markov Analysis Parameters
    # ============================================================================
    N_PERMUTATIONS = int(os.getenv("N_PERMUTATIONS", "1000"))
    RUN_PERMUTATION = os.getenv("RUN_PERMUTATION", "True").lower() == "true"
    RUN_CV = os.getenv("RUN_CV", "True").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate required configuration items"""
        errors = []
        
        # Check required configurations
        if not cls.OPENAI_API_KEY:
            errors.append("❌ OPENAI_API_KEY not set")
        
        # Check file paths
        if cls.TOOL_DESC_FILE:
            tool_desc_path = Path(cls.TOOL_DESC_FILE)
            if not tool_desc_path.is_absolute():
                # Convert relative path to absolute path
                tool_desc_path = cls.PACKAGE_ROOT / tool_desc_path
            if not tool_desc_path.exists():
                errors.append(f"❌ Tool description file does not exist: {tool_desc_path}")
        
        if cls.TRAJECTORY_DIR:
            traj_path = Path(cls.TRAJECTORY_DIR)
            if not traj_path.is_absolute():
                traj_path = cls.PACKAGE_ROOT / traj_path
            if not traj_path.exists():
                errors.append(f"⚠️  Trajectory directory does not exist: {traj_path}")
        
        if errors:
            error_msg = "\n".join(errors)
            print(f"\n{'='*80}")
            print("Configuration validation failed:")
            print(error_msg)
            print(f"{'='*80}\n")
            print("Hints:")
            print(f"  1. Check if {cls.PACKAGE_ROOT}/.env file exists")
            print(f"  2. Refer to {cls.PACKAGE_ROOT}/.env.example configuration template")
            print(f"  3. Or set corresponding environment variables")
            raise ValueError("Configuration validation failed")
        
        # Create output directories
        for dir_path in [cls.OUTPUT_DIR, cls.PIE_CHART_DIR, cls.MARKOV_ANALYSIS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_absolute_path(cls, path: str) -> Path:
        """
        Convert relative path to absolute path
        If already absolute path, return directly
        """
        p = Path(path)
        if p.is_absolute():
            return p
        return cls.PACKAGE_ROOT / p
    
    @classmethod
    def print_config(cls, show_sensitive=False):
        """Print current configuration"""
        print("=" * 80)
        print("📋 AutoTool Configuration")
        print("=" * 80)
        print(f"Package Root: {cls.PACKAGE_ROOT}")
        print(f"\n🔑 API Configuration:")
        print(f"  Base URL: {cls.OPENAI_BASE_URL}")
        if show_sensitive:
            print(f"  API Key: {cls.OPENAI_API_KEY}")
        else:
            print(f"  API Key: {'*' * 20 if cls.OPENAI_API_KEY else '❌ Not set'}")
        
        print(f"\n🤖 Model Configuration:")
        print(f"  Model Name: {cls.MODEL_NAME}")
        print(f"  SimCSE Path: {cls.SIMCSE_MODEL_PATH}")
        print(f"  Temperature: {cls.TEMPERATURE}")
        print(f"  Top P: {cls.TOP_P}")
        
        print(f"\n📁 Data Paths:")
        print(f"  Tool Desc: {cls.TOOL_DESC_FILE}")
        print(f"  Trajectory: {cls.TRAJECTORY_DIR}")
        
        print(f"\n📊 Output Paths:")
        print(f"  Output Dir: {cls.OUTPUT_DIR}")
        print(f"  Pie Charts: {cls.PIE_CHART_DIR}")
        print(f"  Markov Analysis: {cls.MARKOV_ANALYSIS_DIR}")
        
        print(f"\n🔬 Analysis Parameters:")
        print(f"  Permutations: {cls.N_PERMUTATIONS}")
        print(f"  Run Permutation: {cls.RUN_PERMUTATION}")
        print(f"  Run CV: {cls.RUN_CV}")
        print("=" * 80)