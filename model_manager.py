import json
import os
from huggingface_hub import list_models

MODEL_DB_FILE = "helsinki_models.json"

# Map common names to ISO codes used by Helsinki
ISO_MAP = {
    "english": "en", "indonesian": "id", "japanese": "ja", 
    "spanish": "es", "french": "fr", "german": "de", 
    "chinese": "zh", "korean": "ko", "russian": "ru",
    "arabic": "ar", "hindi": "hi", "thai": "th",
    "vietnamese": "vi", "italian": "it", "dutch": "nl"
}

def update_model_db():
    print("\n[SYSTEM] Fetching model list from Hugging Face (One-time process)...")
    print("This might take 10-20 seconds...")
    
    # Fetch all models from author Helsinki-NLP that contain 'opus-mt-'
    models = list_models(author="Helsinki-NLP", search="opus-mt-")
    
    db = []
    for m in models:
        # Format is usually: Helsinki-NLP/opus-mt-{src}-{tgt}
        # Example: Helsinki-NLP/opus-mt-en-id
        try:
            parts = m.modelId.split("-")
            # Usually the last two parts are src and tgt
            # But sometimes it's complex like 'es-en'
            if len(parts) >= 4:
                src = parts[-2]
                tgt = parts[-1]
                db.append({
                    "id": m.modelId,
                    "src": src,
                    "tgt": tgt
                })
        except:
            continue
            
    with open(MODEL_DB_FILE, "w") as f:
        json.dump(db, f)
    
    print(f"[SYSTEM] Database updated! Found {len(db)} Helsinki models.")
    return db

def load_model_db():
    if not os.path.exists(MODEL_DB_FILE):
        return update_model_db()
    
    with open(MODEL_DB_FILE, "r") as f:
        return json.load(f)

def find_helsinki_models(source_lang_name, target_lang_name):
    """
    Returns a list of valid models matching the language pair.
    """
    db = load_model_db()
    
    src_iso = ISO_MAP.get(source_lang_name.lower())
    tgt_iso = ISO_MAP.get(target_lang_name.lower())

    if not src_iso or not tgt_iso:
        return []

    matches = []
    for model in db:
        # Direct match (e.g. en -> id)
        if model['src'] == src_iso and model['tgt'] == tgt_iso:
            matches.append(model)
    
    return matches