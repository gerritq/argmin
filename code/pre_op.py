# # # import os
# # # import json
# # # import re
# # # import torch
# # # from tqdm import tqdm
# # # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # # ==========================================
# # # # 1. Initialization & Keyword Lists
# # # # ==========================================
# # # model_name = "Qwen/Qwen3-8B"
# # # tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
# # # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="")

# # # en_preambular_triggers = [
# # #     "acknowledging", "affirming", "appreciating", "approving", "aware", 
# # #     "bearing in mind", "believing", "commending", "concerned", "conscious", 
# # #     "considering", "convinced", "desiring", "emphasizing", "expecting", 
# # #     "expressing", "fully aware", "guided by", "having adopted", 
# # #     "having considered", "having noted", "having reviewed", "mindful", 
# # #     "noting", "observing", "realising", "recalling", "recognizing", 
# # #     "seeking", "taking into consideration", "underlining", "welcoming", "whereas",
# # #     "while acknowledging"
# # # ]

# # # en_operative_triggers = [
# # #     "accepts", "adopts", "agrees", "appeals", "approves", "authorizes", 
# # #     "calls upon", "commends", "considers", "decides", "declares", 
# # #     "determines", "directs", "draws the attention", "emphasizes", "encourages", "endorses", 
# # #     "expresses appreciation", "expresses hope", "invites", "notes", 
# # #     "proclaims", "reaffirms", "recommends", "reminds", "repeals", 
# # #     "requests", "resolves", "suggests", "supports", "takes note", "urges", "wishes"
# # # ]

# # # fr_preambular_triggers = [
# # #     "reconnaissant", "affirmant", "appréciant", "approuvant", "consciente",
# # #     "gardant à l'esprit", "croyant", "félicitant", "préoccupée", "considérant",
# # #     "convaincue", "désireuse", "soulignant", "s'attendant", "exprimant",
# # #     "pleinement consciente", "guidée par", "ayant adopté", "ayant examiné",
# # #     "ayant noté", "notant", "observant", "réalisant", "rappelant", 
# # #     "tout en admettant"
# # # ]

# # # fr_operative_triggers = [
# # #     "accepte", "adopte", "convient", "lance un appel", "approuve", "autorise",
# # #     "demande", "félicite", "considère", "estime", "décide", "déclare",
# # #     "détermine", "charge", "attire l'attention", "encourage", "appuie", "exprime", 
# # #     "invite", "note", "proclame", "réaffirme", "recommande", "rappelle", 
# # #     "abroge", "prie", "résout", "suggère", "prend note", "prie instamment", 
# # #     "souhaite", "souligne", "constate", "reconnait"
# # # ]

# # # # ==========================================
# # # # 2. Helpers & Classification Function
# # # # ==========================================

# # # def parse_llm_output(raw_text):
# # #     """Safely extracts JSON from LLM output, handling markdown wrappers."""
# # #     try:
# # #         return json.loads(raw_text)
# # #     except json.JSONDecodeError:
# # #         match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
# # #         if match:
# # #             try:
# # #                 return json.loads(match.group(0))
# # #             except:
# # #                 pass
# # #     return {"type": "unknown", "thought": "Failed to parse LLM output."}

# # # def classify_paragraph(para_dict):
# # #     fr_text = para_dict.get("para", "").strip()
# # #     en_text = para_dict.get("para_en", "").strip()
    
# # #     # --- Stage 1: Strict Deterministic Matching ---
# # #     numbering_pattern = r'(?:^\d+[\.\)]\s+|\b\d+[\.\)]\s+)'
# # #     is_numbered = bool(re.search(numbering_pattern, fr_text) or re.search(numbering_pattern, en_text))
    
# # #     ends_with_semicolon = fr_text.endswith(';') or en_text.endswith(';')
# # #     ends_with_period = fr_text.endswith('.') or en_text.endswith('.')
# # #     ends_with_comma = fr_text.endswith(',') or en_text.endswith(',')
    
# # #     fr_lower = fr_text.lower()
# # #     en_lower = en_text.lower()

# # #     starts_with_en_op = any(t in en_lower[:100] for t in en_operative_triggers)
# # #     starts_with_fr_op = any(t in fr_lower[:100] for t in fr_operative_triggers)
# # #     is_operative_keyword = starts_with_en_op or starts_with_fr_op

# # #     starts_with_en_pre = any(t in en_lower[:100] for t in en_preambular_triggers)
# # #     starts_with_fr_pre = any(t in fr_lower[:100] for t in fr_preambular_triggers)
# # #     is_preambular_keyword = starts_with_en_pre or starts_with_fr_pre

# # #     if is_numbered or ((ends_with_semicolon or ends_with_period) and is_operative_keyword):
# # #         return {
# # #             "type": "operative", 
# # #             "method": "deterministic", 
# # #             "reasoning": f"Numbered({is_numbered}), Semicolon/Period({ends_with_semicolon or ends_with_period}), OpKeyword({is_operative_keyword})"
# # #         }

# # #     if not is_numbered and ends_with_comma and is_preambular_keyword:
# # #         return {
# # #             "type": "preambular", 
# # #             "method": "deterministic", 
# # #             "reasoning": f"Comma({ends_with_comma}), PreKeyword({is_preambular_keyword})"
# # #         }

# # #     # --- Stage 2: Probabilistic LLM Fallback ---
# # #     prompt = f"""You are an expert in United Nations resolutions. Classify the paragraph below as strictly 'preambular' or 'operative'.

# # #     Definitions:
# # #     - Preambular Paragraphs: Explain the basis for the action called for. They build an argument, support, or express general principles. They typically begin with a present participle verb (e.g., Noting, Recalling) and end with a comma.
# # #     - Operative Paragraphs: Express what the conference has decided to do. They use clear, actionable language, typically beginning with a present tense verb (e.g., Accepts, Decides) and ending with a semicolon or a period. They are usually numbered unless it is the only operative clause.

# # #     Paragraph to analyze:
# # #     "{en_text}"

# # #     Analyze the syntax and intent. Provide the final result as JSON with exactly two keys: "thought" (your reasoning) and "type" (either "preambular" or "operative")."""
    
# # #     messages = [{"role": "user", "content": prompt}]
# # #     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# # #     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
# # #     generated_ids = model.generate(
# # #         **model_inputs, 
# # #         max_new_tokens=1024,
# # #         temperature=0.4, 
# # #         top_p=0.90,
# # #         do_sample=True
# # #     )
    
# # #     input_length = model_inputs.input_ids.shape[1]
# # #     response_ids = generated_ids[0, input_length:]
# # #     content = tokenizer.decode(response_ids, skip_special_tokens=True)
    
# # #     parsed_llm = parse_llm_output(content)
    
# # #     # Clean up newlines in LLM thought so it formats nicely on a single line
# # #     llm_thought_clean = parsed_llm.get("thought", content).replace('\n', ' ').strip()
    
# # #     return {
# # #         "type": parsed_llm.get("type", "unknown").lower(),
# # #         "method": "LLM",
# # #         "reasoning": llm_thought_clean
# # #     }

# # # # ==========================================
# # # # 3. Batch Processing Logic
# # # # ==========================================

# # # def process_folder(input_folder, output_folder):
# # #     os.makedirs(output_folder, exist_ok=True)
    
# # #     files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
# # #     print(f"Found {len(files)} JSON files. Starting processing...")
    
# # #     for filename in tqdm(files, desc="Processing files"):
# # #         input_path = os.path.join(input_folder, filename)
# # #         output_path = os.path.join(output_folder, filename)
        
# # #         with open(input_path, 'r', encoding='utf-8') as f:
# # #             data = json.load(f)
            
# # #         preambular_list = []
# # #         operative_list = []
# # #         overall_think_log = []
        
# # #         # Process each paragraph
# # #         paragraphs = data.get("body", {}).get("paragraphs", [])
# # #         for para in paragraphs:
# # #             result = classify_paragraph(para)
            
# # #             para_type = result.get("type")
# # #             para_method = result.get("method")
# # #             para_reasoning = result.get("reasoning")
# # #             para_num = para.get("para_number", "Unknown")
            
# # #             # 1. Update individual paragraph object
# # #             para["type"] = para_type
# # #             para["think"] = para_reasoning
            
# # #             # 2. Sort into metadata arrays
# # #             if para_type == "preambular":
# # #                 preambular_list.append(para_num)
# # #             elif para_type == "operative":
# # #                 operative_list.append(para_num)
                
# # #             # 3. Append to the overall metadata think log
# # #             log_entry = f"Paragraph {para_num} - {para_method} - {para_reasoning}"
# # #             overall_think_log.append(log_entry)
                
# # #         # Update metadata structure
# # #         if "METADATA" in data and "structure" in data["METADATA"]:
# # #             data["METADATA"]["structure"]["preambular_para"] = preambular_list
# # #             data["METADATA"]["structure"]["operative_para"] = operative_list
# # #             data["METADATA"]["structure"]["think"] = "\n".join(overall_think_log)
            
# # #         # Save updated file
# # #         with open(output_path, 'w', encoding='utf-8') as f:
# # #             json.dump(data, f, indent=2, ensure_ascii=False)

# # # if __name__ == "__main__":
# # #     # Define your folder paths here
# # #     INPUT_DIR = "./data/test-data"   # Replace with your actual input folder path
# # #     OUTPUT_DIR = "./data/test-data-solved-pre-op" # Replace with your actual output folder path
    
# # #     process_folder(INPUT_DIR, OUTPUT_DIR)
# # #     print("Processing complete!")


# # import os
# # import json
# # import re
# # import torch
# # from tqdm import tqdm
# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # # ==========================================
# # # 1. Initialization & Keyword Lists
# # # ==========================================
# # model_name = "Qwen/Qwen3-8B"
# # tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
# # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="")

# # en_preambular_triggers = [
# #     "acknowledging", "affirming", "appreciating", "approving", "aware", 
# #     "bearing in mind", "believing", "commending", "concerned", "conscious", 
# #     "considering", "convinced", "desiring", "emphasizing", "expecting", 
# #     "expressing", "fully aware", "guided by", "having adopted", 
# #     "having considered", "having noted", "having reviewed", "mindful", 
# #     "noting", "observing", "realising", "recalling", "recognizing", 
# #     "seeking", "taking into consideration", "underlining", "welcoming", "whereas",
# #     "while acknowledging"
# # ]

# # en_operative_triggers = [
# #     "accepts", "adopts", "agrees", "appeals", "approves", "authorizes", 
# #     "calls upon", "commends", "considers", "decides", "declares", 
# #     "determines", "directs", "draws the attention", "emphasizes", "encourages", "endorses", 
# #     "expresses appreciation", "expresses hope", "invites", "notes", 
# #     "proclaims", "reaffirms", "recommends", "reminds", "repeals", 
# #     "requests", "resolves", "suggests", "supports", "takes note", "urges", "wishes"
# # ]

# # fr_preambular_triggers = [
# #     "reconnaissant", "affirmant", "appréciant", "approuvant", "consciente",
# #     "gardant à l'esprit", "croyant", "félicitant", "préoccupée", "considérant",
# #     "convaincue", "désireuse", "soulignant", "s'attendant", "exprimant",
# #     "pleinement consciente", "guidée par", "ayant adopté", "ayant examiné",
# #     "ayant noté", "notant", "observant", "réalisant", "rappelant", 
# #     "tout en admettant"
# # ]

# # fr_operative_triggers = [
# #     "accepte", "adopte", "convient", "lance un appel", "approuve", "autorise",
# #     "demande", "félicite", "considère", "estime", "décide", "déclare",
# #     "détermine", "charge", "attire l'attention", "encourage", "appuie", "exprime", 
# #     "invite", "note", "proclame", "réaffirme", "recommande", "rappelle", 
# #     "abroge", "prie", "résout", "suggère", "prend note", "prie instamment", 
# #     "souhaite", "souligne", "constate", "reconnait"
# # ]

# # # ==========================================
# # # 2. Helpers & Classification Function
# # # ==========================================

# # def parse_llm_output(raw_text):
# #     """Safely extracts JSON from LLM output, handling markdown wrappers."""
# #     try:
# #         return json.loads(raw_text)
# #     except json.JSONDecodeError:
# #         match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
# #         if match:
# #             try:
# #                 return json.loads(match.group(0))
# #             except:
# #                 pass
# #     return {"type": "unknown"}

# # def classify_paragraph(para_dict):
# #     fr_text = para_dict.get("para", "").strip()
# #     en_text = para_dict.get("para_en", "").strip()
    
# #     # --- Stage 1: Strict Deterministic Matching ---
# #     numbering_pattern = r'(?:^\d+[\.\)]\s+|\b\d+[\.\)]\s+)'
# #     is_numbered = bool(re.search(numbering_pattern, fr_text) or re.search(numbering_pattern, en_text))
    
# #     ends_with_semicolon = fr_text.endswith(';') or en_text.endswith(';')
# #     ends_with_period = fr_text.endswith('.') or en_text.endswith('.')
# #     ends_with_comma = fr_text.endswith(',') or en_text.endswith(',')
    
# #     fr_lower = fr_text.lower()
# #     en_lower = en_text.lower()

# #     starts_with_en_op = any(t in en_lower[:100] for t in en_operative_triggers)
# #     starts_with_fr_op = any(t in fr_lower[:100] for t in fr_operative_triggers)
# #     is_operative_keyword = starts_with_en_op or starts_with_fr_op

# #     starts_with_en_pre = any(t in en_lower[:100] for t in en_preambular_triggers)
# #     starts_with_fr_pre = any(t in fr_lower[:100] for t in fr_preambular_triggers)
# #     is_preambular_keyword = starts_with_en_pre or starts_with_fr_pre

# #     if is_numbered or ((ends_with_semicolon or ends_with_period) and is_operative_keyword):
# #         return {
# #             "type": "operative", 
# #             "method": "deterministic", 
# #             "reasoning": f"Numbered({is_numbered}), Semicolon/Period({ends_with_semicolon or ends_with_period}), OpKeyword({is_operative_keyword})"
# #         }

# #     if not is_numbered and ends_with_comma and is_preambular_keyword:
# #         return {
# #             "type": "preambular", 
# #             "method": "deterministic", 
# #             "reasoning": f"Comma({ends_with_comma}), PreKeyword({is_preambular_keyword})"
# #         }

# #     # --- Stage 2: Probabilistic LLM Fallback ---
# #     prompt = f"""You are an expert in United Nations resolutions. Classify the paragraph below as strictly 'preambular' or 'operative'.

# #     Definitions:
# #     - Preambular Paragraphs: The preamble of a draft resolution states the reasons for which the committee is addressing the topic and highlights past international action on the issue. Explain the basis for the action called for. 
# #     - Operative Paragraphs: Operative paragraphs identify the actions or recommendations made in a resolution. Express what the conference has decided to do. 

# #     Paragraph to analyze:
# #     "{en_text}"

# #     Think through the syntax and intent. Then, output your final answer as JSON containing ONLY one key: "type" (either "preambular" or "operative")."""
    
# #     messages = [{"role": "user", "content": prompt}]
# #     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# #     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
# #     generated_ids = model.generate(
# #         **model_inputs, 
# #         max_new_tokens=1024,
# #         temperature=0.4, 
# #         top_p=0.90,
# #         do_sample=True
# #     )
    
# #     input_length = model_inputs.input_ids.shape[1]
# #     response_ids = generated_ids[0, input_length:]
# #     content = tokenizer.decode(response_ids, skip_special_tokens=True)
    
# #     # Extract Native <think> Tags
# #     think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
# #     llm_thought = think_match.group(1).strip().replace('\n', ' ') if think_match else "No thinking extracted."
    
# #     # Parse the remaining JSON
# #     json_only = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
# #     parsed_llm = parse_llm_output(json_only)
    
# #     return {
# #         "type": parsed_llm.get("type", "unknown").lower(),
# #         "method": "LLM",
# #         "reasoning": llm_thought
# #     }

# # # ==========================================
# # # 3. Batch Processing Logic
# # # ==========================================

# # def process_folder(input_folder, output_folder):
# #     os.makedirs(output_folder, exist_ok=True)
    
# #     files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
# #     print(f"Found {len(files)} JSON files. Starting processing...")
    
# #     # Outer progress bar for files
# #     for filename in tqdm(files, desc="Files Processed", position=0):
# #         input_path = os.path.join(input_folder, filename)
# #         output_path = os.path.join(output_folder, filename)
        
# #         with open(input_path, 'r', encoding='utf-8') as f:
# #             data = json.load(f)
            
# #         preambular_list = []
# #         operative_list = []
# #         overall_think_log = []
        
# #         paragraphs = data.get("body", {}).get("paragraphs", [])
        
# #         # Inner progress bar for paragraphs (leave=False clears it when the file is done)
# #         for para in tqdm(paragraphs, desc=f"Paras in {filename}", position=1, leave=False):
# #             result = classify_paragraph(para)
            
# #             para_type = result.get("type")
# #             para_method = result.get("method")
# #             para_reasoning = result.get("reasoning")
# #             para_num = para.get("para_number", "Unknown")
            
# #             # 1. Update individual paragraph object
# #             para["type"] = para_type
# #             # para["think"] = para_reasoning
            
# #             # 2. Sort into metadata arrays
# #             if para_type == "preambular":
# #                 preambular_list.append(para_num)
# #             elif para_type == "operative":
# #                 operative_list.append(para_num)
                
# #             # 3. Append to the overall metadata think log
# #             log_entry = f"Paragraph {para_num} - {para_method} - {para_reasoning}"
# #             overall_think_log.append(log_entry)
                
# #         # Update metadata structure
# #         if "METADATA" in data and "structure" in data["METADATA"]:
# #             data["METADATA"]["structure"]["preambular_para"] = preambular_list
# #             data["METADATA"]["structure"]["operative_para"] = operative_list
# #             data["METADATA"]["structure"]["think"] = "\n".join(overall_think_log)
            
# #         # Save updated file
# #         with open(output_path, 'w', encoding='utf-8') as f:
# #             json.dump(data, f, indent=2, ensure_ascii=False)

# # if __name__ == "__main__":
# #     INPUT_DIR = "./data/test-data"   # Replace with your actual input folder path
# #     OUTPUT_DIR = "./data/test-data-solved-pre-op" # Replace with your actual output folder path
    
# #     process_folder(INPUT_DIR, OUTPUT_DIR)
# #     print("\nProcessing complete!")

# import os
# import json
# import re
# import torch
# from tqdm import tqdm
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # ==========================================
# # 1. Initialization & Model Setup
# # ==========================================
# model_name = "Qwen/Qwen3-8B"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token="")
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="")

# # Bilingual UN Triggers
# en_preambular_triggers = [
#     "acknowledging", "affirming", "appreciating", "approving", "aware", 
#     "bearing in mind", "believing", "commending", "concerned", "conscious", 
#     "considering", "convinced", "desiring", "emphasizing", "expecting", 
#     "expressing", "fully aware", "guided by", "having adopted", 
#     "having considered", "having noted", "having reviewed", "mindful", 
#     "noting", "observing", "realising", "recalling", "recognizing", 
#     "seeking", "taking into consideration", "underlining", "welcoming", "whereas",
#     "while acknowledging"
# ]

# en_operative_triggers = [
#     "accepts", "adopts", "agrees", "appeals", "approves", "authorizes", 
#     "calls upon", "commends", "considers", "decides", "declares", 
#     "determines", "directs", "draws the attention", "emphasizes", "encourages", "endorses", 
#     "expresses appreciation", "expresses hope", "invites", "notes", 
#     "proclaims", "reaffirms", "recommends", "reminds", "repeals", 
#     "requests", "resolves", "suggests", "supports", "takes note", "urges", "wishes"
# ]

# fr_preambular_triggers = [
#     "reconnaissant", "affirmant", "appréciant", "approuvant", "consciente",
#     "gardant à l'esprit", "croyant", "félicitant", "préoccupée", "considérant",
#     "convaincue", "désireuse", "soulignant", "s'attendant", "exprimant",
#     "pleinement consciente", "guidée par", "ayant adopté", "ayant examiné",
#     "ayant noté", "notant", "observant", "réalisant", "rappelant", 
#     "tout en admettant"
# ]

# fr_operative_triggers = [
#     "accepte", "adopte", "convient", "lance un appel", "approuve", "autorise",
#     "demande", "félicite", "considère", "estime", "décide", "déclare",
#     "détermine", "charge", "attire l'attention", "encourage", "appuie", "exprime", 
#     "invite", "note", "proclame", "réaffirme", "recommande", "rappelle", 
#     "abroge", "prie", "résout", "suggère", "prend note", "prie instamment", 
#     "souhaite", "souligne", "constate", "reconnait"
# ]

# # ==========================================
# # 2. Logic Helpers
# # ==========================================

# def parse_llm_json(raw_text):
#     """Extracts JSON from LLM output, handling markdown wrappers."""
#     try:
#         return json.loads(raw_text)
#     except json.JSONDecodeError:
#         match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
#         if match:
#             try: return json.loads(match.group(0))
#             except: pass
#     return {"type": "unknown"}

# def classify_paragraph(para_dict, last_type=None):
#     fr_text = para_dict.get("para", "").strip()
#     en_text = para_dict.get("para_en", "").strip()
#     fr_lower, en_lower = fr_text.lower(), en_text.lower()
    
#     # --- Rule A: Numbering Detection (Handles '1.' or '1)') ---
#     num_pattern = r'(?:^\d+[\.\)]\s+|\b\d+[\.\)]\s+)'
#     is_numbered = bool(re.search(num_pattern, fr_text) or re.search(num_pattern, en_text))

#     # --- Rule B: Continuation Inheritance (That/Que) ---
#     # Strip number if present to check the actual start word
#     clean_en = re.sub(num_pattern, '', en_lower).strip()
#     clean_fr = re.sub(num_pattern, '', fr_lower).strip()
#     is_continuation = clean_en.startswith("that") or clean_fr.startswith("que")
    
#     if is_continuation and last_type:
#         return {
#             "type": last_type, 
#             "method": "inheritance", 
#             "reasoning": f"Clause continuation (That/Que) inheriting from previous {last_type} context."
#         }

#     # --- Rule C: Deterministic Stage (Keyword + Punctuation) ---
#     # Expanded punctuation for Operatives (includes colon for 'Recommends:')
#     ends_with_op_punct = any(fr_text.endswith(p) or en_text.endswith(p) for p in [';', '.', ':'])
#     ends_with_comma = fr_text.endswith(',') or en_text.endswith(',')
    
#     is_op_key = any(t in en_lower[:100] for t in en_operative_triggers) or \
#                 any(t in fr_lower[:100] for t in fr_operative_triggers)
#     is_pre_key = any(t in en_lower[:100] for t in en_preambular_triggers) or \
#                  any(t in fr_lower[:100] for t in fr_preambular_triggers)

#     if is_numbered or (ends_with_op_punct and is_op_key):
#         return {"type": "operative", "method": "deterministic", "reasoning": "Standard Operative numbering or key/punct match."}

#     if ends_with_comma and is_pre_key:
#         return {"type": "preambular", "method": "deterministic", "reasoning": "Standard Preambular keyword/comma match."}

#     # --- Rule D: LLM Fallback (Qwen3 Reasoning) ---
#     prompt = f"""You are an expert in United Nations resolutions. Classify the paragraph below as strictly 'preambular' or 'operative'.
#     Definitions:
#     - Preambular Paragraphs: The preamble of a draft resolution states the reasons for which the committee is addressing the topic and highlights past international action on the issue. Explain the basis for the action called for. 
#     - Operative Paragraphs: Operative paragraphs identify the actions or recommendations made in a resolution. Express what the conference has decided to do. 
#     Paragraph: "{en_text}"
#     Think through the syntax and intent. Then, output your final answer as JSON containing ONLY one key: "type" (either "preambular" or "operative")."""
    
#     messages = [{"role": "user", "content": prompt}]
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
#     generated_ids = model.generate(**model_inputs, max_new_tokens=1024, temperature=0.4, do_sample=True)
#     response_ids = generated_ids[0, model_inputs.input_ids.shape[1]:]
#     content = tokenizer.decode(response_ids, skip_special_tokens=True)
#     print(content)
    
#     # Extract <think> and JSON
#     think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
#     thought = think_match.group(1).strip().replace('\n', ' ') if think_match else "No reasoning extracted."
#     json_only = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
#     return {
#         "type": parse_llm_json(json_only).get("type", "unknown").lower(),
#         "method": "LLM",
#         "reasoning": thought
#     }

# # ==========================================
# # 3. Main Processor
# # ==========================================

# def run_pipeline(in_dir, out_dir):
#     os.makedirs(out_dir, exist_ok=True)
#     files = [f for f in os.listdir(in_dir) if f.endswith('.json')]
    
#     for filename in tqdm(files, desc="Files Progress", position=0):
#         with open(os.path.join(in_dir, filename), 'r', encoding='utf-8') as f:
#             data = json.load(f)
            
#         pre_idx, op_idx, think_log = [], [], []
#         context_type = None  # Resets per file
        
#         paras = data.get("body", {}).get("paragraphs", [])
#         for p in tqdm(paras, desc=f"Processing {filename[:15]}...", position=1, leave=False):
#             res = classify_paragraph(p, last_type=context_type)
#             p_type = res["type"]
            
#             # Update Context for inheritance (only on new heads)
#             if res["method"] != "inheritance" and p_type in ["preambular", "operative"]:
#                 context_type = p_type
            
#             # p["type"], p["think"] = p_type, res["reasoning"]
#             p["type"] = p_type
            
#             if p_type == "preambular": pre_idx.append(p["para_number"])
#             elif p_type == "operative": op_idx.append(p["para_number"])
            
#             think_log.append(f"Para {p.get('para_number')} - {res['method']} - {res['reasoning']}")

#         # Final Metadata Update
#         if "METADATA" in data:
#             struct = data["METADATA"].get("structure", {})
#             struct["preambular_para"] = pre_idx
#             struct["operative_para"] = op_idx
#             struct["think"] = "\n".join(think_log)
#             data["METADATA"]["structure"] = struct
            
#         with open(os.path.join(out_dir, filename), 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)

# if __name__ == "__main__":
#     INPUT_PATH = "./data/test-data"   # Replace with yours
#     OUTPUT_PATH = "./data/test-data-solved-pre-op" # Replace with yours
#     run_pipeline(INPUT_PATH, OUTPUT_PATH)

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