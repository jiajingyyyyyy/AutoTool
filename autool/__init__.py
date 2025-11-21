from .config import Config
from .utils.embedding import ToolEmbeddingService
from .utils.call_model import call_model

__version__ = "0.1.0"
__author__ = "Jingyi Jia"
__all__ = ["Config", "ToolEmbeddingService", "call_model"]

# Optional: Validate configuration on import
# Controlled by environment variable AUTOTOOL_VALIDATE_ON_IMPORT
import os
if os.getenv("AUTOTOOL_VALIDATE_ON_IMPORT", "false").lower() == "true":
    try:
        Config.validate()
        print("✅ AutoTool configuration validated successfully")
    except ValueError as e:
        print(f"⚠️  Configuration validation failed: {e}")
        print(f"Hint: Please check {Config.PACKAGE_ROOT}/.env file")