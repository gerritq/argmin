import os
import json
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# 1. Initialization
# ==========================================
model_name = "Qwen/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="")

# Bilingual Triggers
en_preambular_triggers = [
    "acknowledging", "affirming", "appreciating", "approving", "aware", 
    "bearing in mind", "believing", "commending", "concerned", "conscious", 
    "considering", "convinced", "desiring", "emphasizing", "expecting", 
    "expressing", "fully aware", "guided by", "having adopted", 
    "having considered", "having noted", "having reviewed", "mindful", 
    "noting", "observing", "realising", "recalling", "recognizing", 
    "seeking", "taking into consideration", "underlining", "welcoming", "whereas"
]

en_operative_triggers = [
    "accepts", "adopts", "agrees", "appeals", "approves", "authorizes", 
    "calls upon", "commends", "considers", "decides", "declares", 
    "determines", "directs", "draws the attention", "emphasizes", "encourages", "endorses", 
    "expresses appreciation", "expresses hope", "invites", "notes", 
    "proclaims", "reaffirms", "recommends", "reminds", "repeals", 
    "requests", "resolves", "suggests", "supports", "takes note", "urges", "wishes"
]

fr_preambular_triggers = [
    "reconnaissant", "affirmant", "appréciant", "approuvant", "consciente",
    "gardant à l'esprit", "croyant", "félicitant", "préoccupée", "considérant",
    "convaincue", "désireuse", "soulignant", "s'attendant", "exprimant",
    "pleinement consciente", "guidée par", "ayant adopté", "ayant examiné",
    "ayant noté", "notant", "observant", "réalisant", "rappelant"
]

fr_operative_triggers = [
    "accepte", "adopte", "convient", "lance un appel", "approuve", "autorise",
    "demande", "félicite", "considère", "estime", "décide", "déclare",
    "détermine", "charge", "attire l'attention", "encourage", "appuie", "exprime", 
    "invite", "note", "proclame", "réaffirme", "recommande", "rappelle", 
    "abroge", "prie", "résout", "suggère", "prend note", "prie instamment", 
    "souhaite", "souligne", "constate", "reconnait"
]

# ==========================================
# 2. Logic Helpers
# ==========================================

def parse_llm_json(raw_text):
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
    return {"type": "preambular"} # Default to preambular if parsing fails

def classify_paragraph(para_dict, last_type=None):
    fr_text = para_dict.get("para", "").strip()
    en_text = para_dict.get("para_en", "").strip()
    fr_lower, en_lower = fr_text.lower(), en_text.lower()
    
    # --- Rule 1: First Paragraph Intro (Now strictly Preambular) ---
    intro_keywords = ["adopts the following", "the conference,", "adopte la recommandation", "la conférence,"]
    if para_dict.get("para_number") == 1 and any(x in en_lower or x in fr_lower for x in intro_keywords):
        return {
            "type": "preambular", 
            "method": "deterministic", 
            "reasoning": "First paragraph introductory clause (Preambular)."
        }

    # --- Rule 2: Continuation Inheritance (That/Que) ---
    num_pattern = r'(?:^\d+[\.\)]\s+|\b\d+[\.\)]\s+)'
    is_numbered = bool(re.search(num_pattern, fr_text) or re.search(num_pattern, en_text))
    
    clean_en = re.sub(num_pattern, '', en_lower).strip()
    clean_fr = re.sub(num_pattern, '', fr_lower).strip()
    is_continuation = clean_en.startswith("that") or clean_fr.startswith("que")
    
    if is_continuation and last_type:
        return {
            "type": last_type, 
            "method": "inheritance", 
            "reasoning": f"Inherited {last_type} context from continuation clause (That/Que)."
        }

    # --- Rule 3: Deterministic Stage ---
    ends_with_op_punct = any(fr_text.endswith(p) or en_text.endswith(p) for p in [';', '.', ':'])
    ends_with_comma = fr_text.endswith(',') or en_text.endswith(',')
    
    is_op_key = any(t in en_lower[:100] for t in en_operative_triggers) or \
                any(t in fr_lower[:100] for t in fr_operative_triggers)
    is_pre_key = any(t in en_lower[:100] for t in en_preambular_triggers) or \
                 any(t in fr_lower[:100] for t in fr_preambular_triggers)

    if is_numbered or (ends_with_op_punct and is_op_key):
        return {"type": "operative", "method": "deterministic", "reasoning": "Standard Operative numbering or keyword/punct match."}

    if (ends_with_comma or ends_with_op_punct) and is_pre_key:
        return {"type": "preambular", "method": "deterministic", "reasoning": "Standard Preambular keyword match."}

    # --- Rule 4: LLM Fallback (Truncated Snippet) ---
    snippet = en_text[:1000] 
    prompt = f"""Classify the UN resolution paragraph as 'preambular' or 'operative'. 
    Provide reasoning and then the final JSON with the key "type".
    Text: "{snippet}" """
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**model_inputs, max_new_tokens=1000, temperature=0.4, do_sample=True)
    response_ids = generated_ids[0, model_inputs.input_ids.shape[1]:]
    content = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
    thought = think_match.group(1).strip().replace('\n', ' ') if think_match else "Reasoning extracted from content."
    json_only = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    llm_type = parse_llm_json(json_only).get("type", "preambular").lower()
    if llm_type not in ["preambular", "operative"]: llm_type = "preambular"
    
    return {"type": llm_type, "method": "LLM", "reasoning": thought}

# ==========================================
# 3. Batch Processing Logic
# ==========================================

def run_pipeline(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(in_dir) if f.endswith('.json')]
    
    for filename in tqdm(files, desc="Files Progress", position=0):
        with open(os.path.join(in_dir, filename), 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        pre_idx, op_idx, think_log = [], [], []
        context_type = "preambular" # Start every document with preambular context
        
        paras = data.get("body", {}).get("paragraphs", [])
        for p in tqdm(paras, desc=f"Processing {filename[:15]}", position=1, leave=False):
            res = classify_paragraph(p, last_type=context_type)
            p_type = res["type"]
            
            # Update Context for inheritance (only on new heads)
            if res["method"] != "inheritance":
                context_type = p_type
            
            # p["type"], p["think"] = p_type, res["reasoning"]
            p["type"] = p_type
            
            if p_type == "preambular": pre_idx.append(p["para_number"])
            else: op_idx.append(p["para_number"])
            
            think_log.append(f"Para {p.get('para_number')} - {res['method']} - {res['reasoning']}")

        if "METADATA" in data:
            struct = data["METADATA"].get("structure", {})
            struct["preambular_para"] = pre_idx
            struct["operative_para"] = op_idx
            struct["think"] = "\n".join(think_log)
            
        with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    run_pipeline("./data/test-data", "./data/test-data-solved-pre-op")