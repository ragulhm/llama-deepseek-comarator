import logging
from models.base_model import BaseLanguageModel
from config import ModelConfig

logger = logging.getLogger(__name__)

class DeepSeekModel(BaseLanguageModel):
    def __init__(self, config: ModelConfig):
        logger.info(f"Initializing DeepSeekModel with ID: {config.model_id}")
        super().__init__(config)

    def _load_model_and_tokenizer(self):
        """Loads DeepSeek 70B model and tokenizer."""
        # This will use the common _initialize_model_from_pretrained helper.
        # Specific DeepSeek optimizations or custom loading can be added here if needed.
        super()._initialize_model_from_pretrained()