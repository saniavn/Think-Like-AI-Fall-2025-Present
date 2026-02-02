import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid
import json
from flask import send_file, after_this_request
from datetime import datetime
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
matplotlib.use('Agg')  # Use non-GUI Agg backend for matplotlib
import gensim.downloader as api
import re

app = Flask(__name__)


USER_DATA_DIR = 'user_data'
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

print("Loading Transformer models...")
models = {
    
    'SmolLM-360M': {
        'tokenizer': AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct'),
        'model': AutoModelForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-360M-Instruct',
                                                    torch_dtype=torch.float16) # Use float16 for speed
    }
    #,
    # 'Qwen2.5-1.5B': {
    #     'tokenizer': AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct'),
    #     'model': AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
    #                                                  torch_dtype=torch.float16,
    #                                                  trust_remote_code=True) # Qwen models require trust_remote_code
    # }
}

print("Transformer models loaded.")

#Root route for mode selection
@app.route("/")
def start_page():
    return render_template("start.html")

#Auto-login route
@app.route("/auto_login")
def auto_login():
    """
    This logic automatically creates a new 'guest' user session
    and redirects to the main application.
    """
    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "first_name": "Guest",
        "last_initial": "User",
        "grade": "N/A",
        "interactions": []
    }

    # Save the new user's JSON file
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)

    # Send the user to the main app page
    return redirect(url_for("home", user_id=user_id))


@app.route("/login", methods=["GET", "POST"])
def study_login():
    if request.method == "POST":
        student_id = request.form.get("student_id", "").strip()
        q1 = request.form.get("q1", "") # Get the new question

        user_id = str(uuid.uuid4())
        user_data = {
            "user_id": user_id,
            "student_id": student_id,
            "pre_survey": { # Store pre-survey answers
                "q1_used_chatbot": q1
            },
            "interactions": []
        }

        user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=4)
        return redirect(url_for("home", user_id=user_id))
    return render_template("login.html")

@app.route("/home")
def home():
    user_id = request.args.get("user_id")
    # Redirect to new study_login route
    if not user_id: return redirect(url_for("study_login"))
    return render_template("index.html", user_id=user_id)
def clean_token(token):
    special_tokens = ['</s>', '<pad>', '<|endoftext|>', '<unk>', '<|imend|>', '<|im_end|>', '<|im_start|>', 'Ċ', 'ĉ', '.*', '."']
    
    if token in special_tokens:
        return None

    cleaned_token = ""
    
    if token.startswith('Ġ'):
        cleaned_token = " " + token[1:] # GPT-2 style: turns "Ġcat" into " cat"
    elif token.startswith(' '): 
        cleaned_token = " " + token[1:] # Llama/SmolLM style: turns " cat" into " cat"
    elif token.startswith('_'):
        cleaned_token = " " + token[1:] # T5 style: turns "_cat" into " cat"
    
    else:
        cleaned_token = token 
    
   
    if not cleaned_token:
        return None
        
    return cleaned_token


def get_next_word_predictions(model_name, text, top_k=20, temperature=1.0, p_value=0.0):
    if model_name not in models:
        print(f"Error: Model {model_name} not found!")
        return [], [], []

    tokenizer = models[model_name]['tokenizer']
    model = models[model_name]['model']
    
    # Encode input
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Ensure temperature is valid (not zero)
    # We treat any temp below 0.01 as 0.01 to prevent division by zero errors
    current_temp = max(float(temperature), 0.01)

    if 'flan-t5' in model_name:
        # T5 logic (Sequence-to-Sequence)
        outputs = model.generate(
            input_ids,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False  # We turn off sampling to get raw logits
        )
        if outputs.scores:
            logits = outputs.scores[0][0]
        else:
            return [], [], []
    else:
        # Causal LM logic (SmolLM, GPT, etc.)
        with torch.no_grad():
            outputs = model(input_ids)
        # Get logits for the last token
        logits = outputs.logits[0, -1, :]

    top_logits, top_indices = torch.topk(logits, top_k)

   
    scaled_top_logits = top_logits / current_temp


    top_probs = torch.softmax(scaled_top_logits, dim=-1)

    # Convert IDs to actual words (tokens)
    top_tokens = tokenizer.convert_ids_to_tokens(top_indices.tolist())

    print(f"RAW {model_name} TOKENS: {top_tokens}")

   
    cleaned_tokens, cleaned_probs, cleaned_ids = [], [], []
    for token, prob, token_id in zip(top_tokens, top_probs.tolist(), top_indices.tolist()):
        cleaned = clean_token(token)
        if cleaned:
            cleaned_tokens.append(cleaned)
            visual_prob = prob if prob > 0.01 else 0.01 
            cleaned_probs.append(visual_prob)
            cleaned_ids.append(token_id)

    # Fallback if empty
    if not cleaned_tokens and top_tokens:
        raw_top_token = top_tokens[0].replace(' ', ' ').strip()
        cleaned_tokens.append(raw_top_token if raw_top_token else top_tokens[0])
        cleaned_probs.append(top_probs[0].item())
        cleaned_ids.append(top_indices[0].item())
    
    # DEBUG PRINT
    for t, p in zip(cleaned_tokens[:5], cleaned_probs[:5]):
        print(f"Token: '{t}' | Probability: {p:.6f}")

    return cleaned_tokens, cleaned_probs, cleaned_ids


@app.route("/chat")
def chat():
    user_id = request.args.get("user_id")
    if not user_id: return redirect(url_for("study_login"))
    return render_template('chat.html', user_id=user_id)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    user_id = data.get('user_id', 'unknown_user')
    model_name = data['model']
    text = data['text']
    temperature = float(data.get('temperature', 1.0))
    p_value = float(data.get('p_value', 0.0))

    
    predicted_tokens, probabilities, predicted_token_ids = get_next_word_predictions(
        model_name, text, 20, 1.0, p_value 
    )

    # We still save the REAL temperature to the log file for the data
    save_user_interaction(
        user_id, "typed", text, model_name, temperature, temperature, p_value, 42,
        predicted_tokens, probabilities, selection_method=data.get("input_source", "typed")
    )

    return jsonify({
        'predicted_tokens': predicted_tokens,
        'probabilities': probabilities,
        'predicted_token_ids': predicted_token_ids
    })
def save_user_interaction(user_id, interaction_type, input_text, model, temperature, temp_slider, p_value, seed,
                          predicted_tokens=None, predicted_probs=None, selected_word=None, selection_method="typed"):
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            user_data = json.load(f)
    else:
        user_data = {"user_id": user_id, "interactions": []}

    # Track prediction counts
    if "prediction_counts" not in user_data:
        user_data["prediction_counts"] = {"chart_clicks": 0, "spin_clicks": 0}

    # Increment the total counts based on the selection method from the frontend
    if selection_method == 'clicked':
        user_data["prediction_counts"]["chart_clicks"] = user_data["prediction_counts"].get("chart_clicks", 0) + 1
    elif selection_method == 'auto-selected':
        user_data["prediction_counts"]["spin_clicks"] = user_data["prediction_counts"].get("spin_clicks", 0) + 1


    interaction = {
        "timestamp": datetime.now().isoformat(),
        "interaction_type": interaction_type,
        "selection_method": selection_method,
        "input_text": input_text,
        "model": model,
        "temperature": temperature,
        "chart_temperature": temp_slider,
        "p_value": p_value,
        "seed": seed,
        "selected_word": selected_word,
        "suggested_words": [{"word": word, "probability": prob} for word, prob in zip(predicted_tokens, predicted_probs)] if predicted_tokens else []
    }

    if "interactions" not in user_data:
        user_data["interactions"] = []
    user_data["interactions"].append(interaction)

    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)



@app.route('/download_user_data/<user_id>')
def download_user_data(user_id):
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if os.path.exists(user_file):
        return send_file(user_file, as_attachment=True)
    return "User data not found", 404


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        model_name = data.get("model")
        query = data.get("query", "")
        temperature = max(float(data.get("temperature", 1.0)), 0.01)
        top_p_value = max(0.01, min(float(data.get("top_p", 1.0)), 1.0)) # Ensure valid top_p

        if not model_name or model_name not in models:
             # Fallback to a default model if none specified or invalid
            model_name = 'SmolLM-360M' # Or choose another default
            print(f"Warning: Model name invalid or not provided. Falling back to {model_name}")

        tokenizer = models[model_name]['tokenizer']
        model = models[model_name]['model']

        if 'flan-t5' in model_name:
            prompt = f"Answer the following question clearly, in several sentences, and provide an explanation:\n\n{query}"
            print(f"Using prompt for {model_name}: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                min_length=100,
                do_sample=True,
                top_p=top_p_value,
                temperature=temperature,
                repetition_penalty=1.2
            )
            output_ids = outputs[0]

        #UPDATED LOGIC FOR CAUSAL LM PROMPTS
        elif model_name in ['TinyLlama-1.1B', 'SmolLM-360M', 'Qwen2.5-1.5B']:
            # Use chat template for modern instruct/chat models
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ]
            # Ensure tokenizer handles chat template correctly, don't add generation prompt if it does it automatically
            try:
                 prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception as e:
                 print(f"Warning: Could not apply chat template for {model_name}. Falling back to basic prompt. Error: {e}")
                 # Fallback basic prompt if template fails
                 prompt = f"System: You are a helpful assistant.\nUser: {query}\nAssistant:"

            print(f"Using prompt for {model_name}: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=100, # Keep chat responses concise
                do_sample=True,
                temperature=temperature,
                top_p=top_p_value,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id, # Handle missing eos_token_id
            )
            # Decode only the generated part
            output_ids = outputs[0][inputs['input_ids'].shape[1]:]
        else:
            # Original logic for older GPT models (gpt2, gpt-neo)
            prompt = f"Q: {query}\nA:"
            print(f"Using prompt for {model_name}: '{prompt}'")
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=inputs['input_ids'].shape[1] + 100, # Original max_length
                do_sample=True,
                temperature=temperature,
                top_p=top_p_value,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
            )
            # Decode only the generated part
            output_ids = outputs[0][inputs['input_ids'].shape[1]:]

        answer = tokenizer.decode(output_ids, skip_special_tokens=True)
        # Simplified cleaning - remove potential instruction artifacts if needed
        final_answer = answer.strip()

        # Logging interaction
        user_id = data.get("user_id")
        if user_id:
            user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
            if os.path.exists(user_file):
                with open(user_file, "r") as f:
                    user_data = json.load(f)

                interaction = {
                    "timestamp": datetime.now().isoformat(),
                    "interaction_type": "chat",
                    "input_text": query,
                    "model": model_name,
                    "response": final_answer,
                    "temperature": temperature,
                    "p_value": top_p_value
                }
                user_data["interactions"].append(interaction)

                with open(user_file, "w") as f:
                    json.dump(user_data, f, indent=4)

        return jsonify({"answer": final_answer})

    except Exception as e:
        import traceback
        print("Error in /ask route:\n", traceback.format_exc())
        return jsonify({"error": "Server error"}), 500

@app.route("/save_questions", methods=["POST"])
def save_questions():
    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "User not found"}), 404

    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if not os.path.exists(user_file):
        return jsonify({"error": "User file not found"}), 404

    with open(user_file, "r") as f:
        user_data = json.load(f)

    # Saving "q5"
    q5 = data.get("q5", "").strip()
    if q5:
        user_data["q5"] = q5

    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)

    return jsonify({"status": "saved"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)