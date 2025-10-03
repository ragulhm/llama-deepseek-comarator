from flask import Flask, render_template, request, session, redirect, url_for, flash
import logging
import os

from config import app_config # Our structured configuration
from models.llama_model import LlamaModel
from models.deepseek_model import DeepSeekModel

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, app_config.LOG_LEVEL.upper()),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

app = Flask(__name__)
app.config['SECRET_KEY'] = app_config.SECRET_KEY # Set Flask secret key
app.config['DEBUG'] = (app_config.LOG_LEVEL.upper() == 'DEBUG')

# Global model instances
llama_model_instance = None
deepseek_model_instance = None

def load_models_globally():
    global llama_model_instance, deepseek_model_instance
    if llama_model_instance is None:
        try:
            logger.info("Attempting to load Llama model...")
            llama_model_instance = LlamaModel(app_config.LLAMA_MODEL_CONFIG)
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
          
            llama_model_instance = None # Ensure it's None if loading fails

    if deepseek_model_instance is None:
        try:
            logger.info("Attempting to load DeepSeek model...")
            deepseek_model_instance = DeepSeekModel(app_config.DEEPSEEK_MODEL_CONFIG)
        except Exception as e:
            logger.error(f"Failed to load DeepSeek model: {e}")
            #flash(f"Error loading DeepSeek model: {e}", "danger")
            deepseek_model_instance = None # Ensure it's None if loading fails

# Call this once when the app starts
with app.app_context():
    load_models_globally()

@app.route('/')
def index():
    return render_template('index.html', 
                           llama_config=app_config.LLAMA_MODEL_CONFIG,
                           deepseek_config=app_config.DEEPSEEK_MODEL_CONFIG)

@app.route('/compare', methods=['POST'])
def compare_models():
    prompt = request.form['prompt']
    if not prompt:
        flash("Please enter a prompt.", "warning")
        return redirect(url_for('index'))

    # Allow overriding model parameters from the form
    try:
        llama_params = {
            'max_new_tokens': int(request.form.get('llama_max_new_tokens', app_config.LLAMA_MODEL_CONFIG.max_new_tokens)),
            'temperature': float(request.form.get('llama_temperature', app_config.LLAMA_MODEL_CONFIG.temperature)),
            'top_p': float(request.form.get('llama_top_p', app_config.LLAMA_MODEL_CONFIG.top_p))
        }
        deepseek_params = {
            'max_new_tokens': int(request.form.get('deepseek_max_new_tokens', app_config.DEEPSEEK_MODEL_CONFIG.max_new_tokens)),
            'temperature': float(request.form.get('deepseek_temperature', app_config.DEEPSEEK_MODEL_CONFIG.temperature)),
            'top_p': float(request.form.get('deepseek_top_p', app_config.DEEPSEEK_MODEL_CONFIG.top_p))
        }
    except ValueError as e:
        flash(f"Invalid parameter value: {e}. Please check your inputs.", "danger")
        return redirect(url_for('index'))


    results = {}
    
    # Llama 3.3 Inference
    if llama_model_instance:
        # Temporarily update config for this request
        original_llama_config = llama_model_instance.config.model_copy()
        llama_model_instance.config.max_new_tokens = llama_params['max_new_tokens']
        llama_model_instance.config.temperature = llama_params['temperature']
        llama_model_instance.config.top_p = llama_params['top_p']

        llama_result = llama_model_instance.generate_response(prompt)
        results['model1'] = {
            'name': llama_result.get('model_name', app_config.LLAMA_MODEL_CONFIG.model_id),
            'response': llama_result['response'],
            'time': f"{llama_result['response_time']:.2f} seconds",
            'speed': f"{llama_result['token_speed']:.2f} tokens/second",
            'tokens_generated': llama_result.get('tokens_generated', 'N/A')
        }
        llama_model_instance.config = original_llama_config # Revert config
        logger.info(f"Llama model ({llama_model_instance.config.model_id}) response generated.")
    else:
        results['model1'] = {
            'name': app_config.LLAMA_MODEL_CONFIG.model_id,
            'response': "Llama model not loaded or encountered an error during initialization.",
            'time': "N/A", 'speed': "N/A", 'tokens_generated': 'N/A'
        }

    # DeepSeek 70B Inference
    if deepseek_model_instance:
        # Temporarily update config for this request
        original_deepseek_config = deepseek_model_instance.config.model_copy()
        deepseek_model_instance.config.max_new_tokens = deepseek_params['max_new_tokens']
        deepseek_model_instance.config.temperature = deepseek_params['temperature']
        deepseek_model_instance.config.top_p = deepseek_params['top_p']

        deepseek_result = deepseek_model_instance.generate_response(prompt)
        results['model2'] = {
            'name': deepseek_result.get('model_name', app_config.DEEPSEEK_MODEL_CONFIG.model_id),
            'response': deepseek_result['response'],
            'time': f"{deepseek_result['response_time']:.2f} seconds",
            'speed': f"{deepseek_result['token_speed']:.2f} tokens/second",
            'tokens_generated': deepseek_result.get('tokens_generated', 'N/A')
        }
        deepseek_model_instance.config = original_deepseek_config # Revert config
        logger.info(f"DeepSeek model ({deepseek_model_instance.config.model_id}) response generated.")
    else:
        results['model2'] = {
            'name': app_config.DEEPSEEK_MODEL_CONFIG.model_id,
            'response': "DeepSeek model not loaded or encountered an error during initialization.",
            'time': "N/A", 'speed': "N/A", 'tokens_generated': 'N/A'
        }
    
    return render_template('results.html', prompt=prompt, results=results)

if __name__ == '__main__':
    # Ensure environment variables are set before running:
    # FLASK_SECRET_KEY="your_secure_secret_key"
    # HF_TOKEN="hf_YOUR_HUGGINGFACE_TOKEN"
    # LLAMA_MODEL_ID="meta-llama/Llama-2-7b-chat-hf" (or Llama 3.3 if available)
    # DEEPSEEK_MODEL_ID="deepseek-ai/deepseek-llm-7b-chat" (or DeepSeek 70B if available)
    # python app.py
    app.run(debug=app_config.LOG_LEVEL.upper() == 'DEBUG', host='0.0.0.0', port=5000)