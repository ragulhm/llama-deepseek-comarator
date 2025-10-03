import logging
from models.base_model import BaseLanguageModel
from config import ModelConfig

logger = logging.getLogger(__name__)

class LlamaModel(BaseLanguageModel):
    def __init__(self, config: ModelConfig):
        logger.info(f"Initializing LlamaModel with ID: {config.model_id}")
        super().__init__(config)

    def _load_model_and_tokenizer(self):
        """Loads Llama 3.3 model and tokenizer."""
        # This will use the common _initialize_model_from_pretrained helper
        # provided in the BaseLanguageModel.
        # Specific Llama optimizations or custom loading can be added here if needed.
        super()._initialize_model_from_pretrained()