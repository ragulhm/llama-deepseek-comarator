from abc import ABC, abstractmethod
import time
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config import ModelConfig # Import ModelConfig
from transformers import AutoTokenizer
# ...
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-chat", force_download=True)

logger = logging.getLogger(__name__)

class BaseLanguageModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model_and_tokenizer()

    @abstractmethod
    def _load_model_and_tokenizer(self):
        """Abstract method to load model and tokenizer, to be implemented by subclasses."""
        pass

    def _initialize_model_from_pretrained(self):
        """Helper to initialize common Hugging Face models."""
        try:
            logger.info(f"Loading tokenizer for {self.config.model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, token=self.config.hf_token)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token # Essential for batching/generation

            logger.info(f"Loading model for {self.config.model_id} on device: {self.device}...")

            # Quantization configuration for efficient loading
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            ) if self.device == "cuda" else None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=bnb_config,
                device_map="auto" if self.device == "cuda" else None,
                token=self.config.hf_token,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32 # Use bfloat16 on CUDA
            )
            self.model.eval() # Set model to evaluation mode
            logger.info(f"Successfully loaded {self.config.model_id}")
        except Exception as e:
            logger.error(f"Error loading model {self.config.model_id}: {e}")
            self.tokenizer = None
            self.model = None
            raise # Re-raise to indicate a critical loading failure

    def generate_response(self, prompt: str) -> dict:
        """
        Generates a response from the model and collects performance metrics.
        Returns a dictionary with response and metrics.
        """
        if not self.model or not self.tokenizer:
            return {
                "response": "Error: Model not loaded or initialized.",
                "response_time": 0,
                "token_speed": 0,
                "model_name": self.config.model_id
            }

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
            input_length = inputs.input_ids.shape[1]

            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_time = time.time()

            response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            response_time = end_time - start_time
            tokens_generated = len(outputs[0]) - input_length
            token_generation_speed = tokens_generated / response_time if response_time > 0 else 0

            return {
                "response": response,
                "response_time": response_time,
                "token_speed": token_generation_speed,
                "model_name": self.config.model_id,
                "tokens_generated": tokens_generated
            }
        except Exception as e:
            logger.error(f"Error during inference for {self.config.model_id}: {e}")
            return {
                "response": f"Error generating response: {e}",
                "response_time": 0,
                "token_speed": 0,
                "model_name": self.config.model_id
            }