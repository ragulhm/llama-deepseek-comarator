# llama-deepseek-comparator-v2/config.py

import os
from pydantic import BaseModel, Field, ValidationError
from typing import Optional
import logging

# --- ADD THIS LINE AT THE VERY TOP ---
from dotenv import load_dotenv
load_dotenv() # Load environment variables from .env file
# -------------------------------------

logger = logging.getLogger(__name__)

class ModelConfig(BaseModel):
    model_id: str
    max_new_tokens: int = 8000
    temperature: float = 0.7
    top_p: float = 0.9
    hf_token: Optional[str] = None # Hugging Face token for private/gated models

class AppConfig(BaseModel):
    SECRET_KEY: str = Field(..., env='FLASK_SECRET_KEY') # Required for Flask sessions/security
    LLAMA_MODEL_CONFIG: ModelConfig
    DEEPSEEK_MODEL_CONFIG: ModelConfig

    # General app settings
    LOG_LEVEL: str = "INFO"

    @classmethod
    def load_from_env(cls):
        try:
            # Construct ModelConfig instances from environment variables
            # os.environ.get() will now correctly read values from your .env file
            llama_config = ModelConfig(
                model_id=os.environ.get('LLAMA_MODEL_ID', "meta-llama/Llama-2-7b-chat-hf"),
                max_new_tokens=int(os.environ.get('LLAMA_MAX_NEW_TOKENS', '200')),
                temperature=float(os.environ.get('LLAMA_TEMPERATURE', '0.7')),
                top_p=float(os.environ.get('LLAMA_TOP_P', '0.9')),
                hf_token=os.environ.get('HF_TOKEN') # Using the HF_TOKEN from .env
            )
            deepseek_config = ModelConfig(
                model_id=os.environ.get('DEEPSEEK_MODEL_ID', "deepseek-ai/deepseek-llm-7b-chat"),
                max_new_tokens=int(os.environ.get('DEEPSEEK_MAX_NEW_TOKENS', '200')),
                temperature=float(os.environ.get('DEEPSEEK_TEMPERATURE', '0.7')),
                top_p=float(os.environ.get('DEEPSEEK_TOP_P', '0.9')),
                hf_token=os.environ.get('HF_TOKEN') # Using the HF_TOKEN from .env
            )

            return cls(
                SECRET_KEY=os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_change_me'), # Reads from .env
                LLAMA_MODEL_CONFIG=llama_config,
                DEEPSEEK_MODEL_CONFIG=deepseek_config,
                LOG_LEVEL=os.environ.get('LOG_LEVEL', 'INFO') # Reads from .env
            )
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

# Initialize and expose config globally for the app
try:
    app_config = AppConfig.load_from_env()
except ValidationError as e:
    logger.critical("Failed to load application configuration. Exiting.")
    exit(1)