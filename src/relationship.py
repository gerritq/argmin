import os
import json
import re
import math
import torch
from tqdm import tqdm  # Added import
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer

class UNResolutionPipeline:
    def __init__(self, input_dir, output_dir, qwen_model_id="Qwen/Qwen3-8B"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Loading SentenceTransformer for Phase 1...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        print(f"Loading {qwen_model_id} for Phase 3...")
        self.tokenizer = AutoTokenizer.from_pretrained(qwen_model_id, token="")
        
        # --- NEW BATCHING CONFIGURATION ---
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            qwen_model_id,
            device_map="auto",
            token=""
        )

    def compute_asymmetric_decay(self, anchor_idx, candidate_idx):
        dist = anchor_idx - candidate_idx
        if dist > 0:
            return math.exp(-0.05 * dist)
        elif dist < 0:
            return math.exp(-0.5 * abs(dist))
        return 0.0

    def get_regex_candidates(self, text):
        if re.search(r'(resolution|document|\b[12][0-9]{3}\b)', text, re.IGNORECASE):
            return []
        matches = re.findall(r'paragraph[s]?\s+(\d+[a-z]?)', text, re.IGNORECASE)
        return [int(re.sub(r'[^0-9]', '', m)) for m in matches if re.sub(r'[^0-9]', '', m).isdigit()]

    def phase_1_filter(self, anchor_para, all_paras):
        anchor_idx = anchor_para.get("para_number")
        if anchor_idx is None:
            return []
            
        anchor_type = anchor_para.get("type")
        anchor_text = anchor_para.get("para_en", "")
        if not anchor_text:
            return []
            
        anchor_emb = self.embedder.encode(anchor_text, convert_to_tensor=True)
        candidates = []
        regex_matches = self.get_regex_candidates(anchor_text)

        for cand in all_paras:
            cand_idx = cand.get("para_number")
            cand_text = cand.get("para_en", "")
            
            if cand_idx is None or cand_idx == anchor_idx or not cand_text:
                continue
                
            cand_type = cand.get("type")
            
            # Hard logic gate
            decay_multiplier = self.compute_asymmetric_decay(anchor_idx, cand_idx)

            if anchor_type == "preambular" and cand_type == "preambular":
                # continue 
                decay_multiplier *= 0.1

            cand_emb = self.embedder.encode(cand_text, convert_to_tensor=True)
            cos_sim = util.cos_sim(anchor_emb, cand_emb).item()
            
            final_score = cos_sim * decay_multiplier
            
            if cand_idx in regex_matches:
                final_score += 1.0 

            candidates.append((cand_idx, cand_text, cand_type, final_score))

        candidates.sort(key=lambda x: x[3], reverse=True)
        # tqdm.write(candidates)
        return candidates[:10]

    def phase_3_llm_disambiguation(self, anchor_para, top_candidates):
        if not top_candidates:
            return {}, {}

        anchor_idx = anchor_para["para_number"]
        anchor_text = anchor_para["para_en"]

        matched_dict = {}
        think_logs = {}
        allowed_relations = {"contradictive", "supporting", "complemental", "modifying"}

        # (Your existing sys_prompt remains exactly the same here)
        sys_prompt = (
            "You are mapping the internal logic of a single, isolated UN document. "
            "You will be given an Anchor Paragraph and a Candidate Paragraph. "
            "Determine the relationship between the Candidate and the Anchor. "
            "Output ONLY a JSON object in this exact format: {\"relation\": \"label\"}. "
            "The label must be one of: 'contradictive', 'supporting', 'complemental', 'modifying', or 'none'.\n\n"
            "SUPPORTING: Candidate provides justification, evidence, or context making Anchor's directive valid.\n"
            "COMPLEMENTAL: Candidate addresses the same theme as Anchor and adds additional info, without depending on each other.\n"
            "MODIFYING: Candidate changes, qualifies, restricts, or expands the scope of the Anchor.\n"
            "CONTRADICTIVE: Candidate asserts something conflicting with the Anchor.\n"
            "NONE: The paragraphs are not related.\n\n"
            "Counterfactual test: Does the existence of the Candidate amend, restrict, or expand "
            "the specific mandate established in the Anchor?\n"
            "- If YES -> 'modifying'.\n"
            "- If NO (it adds a related but separate action) -> 'complemental'.\n\n"
            "Provide your thinking process in <think>...</think> tags, then output the final JSON.\n"
            "If the value is none, no need to output it.\n"
            "Here are some examples of paragraphs and their relation\n"
            "Example 1\n"
            "Paragraph A: Considering that a certain number of students admitted to secondary schools are not in a position to benefit effectively from the instruction provided therein;\n"
            "Paragraph B: Deems it necessary, in order to avoid as much as possible errors in orientation and the discouragement that may result, to organize student guidance during the final regulated year of primary education, with the collaboration of the teacher, the physician, and the vocational guidance service, with the decision remaining the responsibility of the family.\n"
            "Relation: supporting\n"
            "Example 2\n"
            "Paragraph A: Considers desirable greater coordination between primary education and secondary education in order to facilitate, especially during the initial years of study, the easy transition from one category of education to another.\n"
            "Paragraph B: Deems it necessary, in order to avoid as much as possible errors in orientation and the discouragement that may result, to organize student guidance during the final regulated year of primary education, with the collaboration of the teacher, the physician, and the vocational guidance service, with the decision remaining the responsibility of the family.\n"
            "Relation: complemental\n"
            "Example 3\n"
            "Paragraph A: Considers it desirable to improve the selection methods for admission to secondary schools proper. For this selection, the following elements should be taken into account: a) the primary school leaving certificate, as well as the individual report prepared by the primary school teachers, b) an examination conducted according to scientific methods aimed at identifying not only the knowledge acquired but also the candidate's aptitude to continue their studies.\n"
            "Paragraph B: Draws the attention of educational authorities to the fact that, since any selection involves forced elimination, any student excluded from the secondary schools proper should be directed towards other studies or practical vocational training corresponding to their aptitudes.\n"
            "Relation: modifying\n"
            "Example 4\n"
            "Paragraph A: Indigenous peoples have the right to self-determination. By virtue of that right they freely determine their political status and freely pursue their economic, social and cultural development.\n"
            "Paragraph B: Nothing in this Declaration may be construed as authorizing or encouraging any action which would dismember or impair, totally or in part, the territorial integrity or political unity of sovereign and independent States.\n"
            "Relation: contradictive\n"
        ) 

        # 1. Prepare all prompts in a single batch
        prompts = []
        candidate_indices = []

        for cand_idx, cand_text, _, _ in top_candidates:
            user_prompt = (
                f"Anchor Paragraph {anchor_idx}:\n{anchor_text}\n\n"
                f"Candidate Paragraph {cand_idx}:\n{cand_text}\n"
            )

            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            candidate_indices.append(cand_idx)

        # 2. Tokenize the batch (Note: padding=True is required now)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.llm.device)
        
        # 3. Generate in parallel
        outputs = self.llm.generate(
            **inputs, 
            max_new_tokens=1024, 
            temperature=0.6,
            top_p=0.95,
            do_sample=True
        )
        
        # 4. Parse the batch outputs
        # The length of the input sequence so we can slice out just the new generated tokens
        input_length = inputs.input_ids.shape[1]

        for i, output in enumerate(outputs):
            cand_idx = candidate_indices[i]
            
            # Decode only the newly generated tokens
            response = self.tokenizer.decode(output[input_length:], skip_special_tokens=True)
            
            # Extract Thinking Process
            think_text = ""
            think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
            if think_match:
                think_text = think_match.group(1).strip()
            
            if think_text:
                think_logs[cand_idx] = think_text
            
            # Extract and Validate JSON
            clean_response = response.replace(f"<think>{think_text}</think>", "").strip()
            # json_match = re.search(r'\{.*\}', clean_response, re.DOTALL)

            # New, non-greedy regex
            json_match = re.search(r'\{.*?\}', clean_response, re.DOTALL)
            
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    relation = str(parsed_json.get("relation", "")).lower().strip()
                    
                    if relation in allowed_relations:
                        matched_dict[cand_idx] = relation
                        
                except Exception as e:
                    tqdm.write(f"JSON Parsing Error on Anchor {anchor_idx} vs Cand {cand_idx}: {e}")

        return matched_dict, think_logs

    def process_all_files(self):
        # Pre-fetch filenames
        filenames = [f for f in os.listdir(self.input_dir) if f.endswith(".json")]
        
        # Outer progress bar: Overall file processing
        for filename in tqdm(filenames, desc="Files Processed", position=0):
            in_path = os.path.join(self.input_dir, filename)
            out_path = os.path.join(self.output_dir, filename)
            lock_path = out_path + ".lock"
            
            # 1. Skip if already fully processed
            if os.path.exists(out_path):
                # Optional: comment this out if it spams your terminal too much
                tqdm.write(f"Skipping {filename}: Already solved.")
                continue

            # 2. Attempt to acquire an atomic file lock
            try:
                # O_CREAT creates the file, O_EXCL ensures it fails if the file already exists
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd) # Close the file descriptor immediately, we just need the file to exist
            except FileExistsError:
                # Another terminal instance beat us to it and is currently processing this file
                tqdm.write(f"Skipping {filename}: Locked by another process.")
                continue
                
            tqdm.write(f"--- Processing File: {filename} ---") 
            
            try:
                # --- START OF PROCESSING LOGIC ---
                with open(in_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    
                paras = doc.get("body", {}).get("paragraphs", [])
                
                # Inner progress bar: Paragraphs within the current file
                for i, anchor in enumerate(tqdm(paras, desc="Paragraphs", position=1, leave=False)):
                    top_candidates = self.phase_1_filter(anchor, paras)
                    matched, think_log = self.phase_3_llm_disambiguation(anchor, top_candidates)
                    
                    doc["body"]["paragraphs"][i]["matched_pars"] = matched
                    doc["body"]["paragraphs"][i]["think"] = think_log
                    
                # Save to new folder with the same filename
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(doc, f, indent=2, ensure_ascii=False)
                
                tqdm.write(f"Successfully saved to: {out_path}\n")
                # --- END OF PROCESSING LOGIC ---

            except Exception as e:
                tqdm.write(f"Error processing {filename}: {e}")
            finally:
                # 3. Clean up the lock file whether processing succeeded or failed
                if os.path.exists(lock_path):
                    os.remove(lock_path)

# Example Execution
if __name__ == "__main__":
    # Define your folders here
    INPUT_FOLDER = "./data/test-data-solved-pre-op"
    OUTPUT_FOLDER = "./data/test-data-solved-pre-op-and-relationship"
    
    pipeline = UNResolutionPipeline(input_dir=INPUT_FOLDER, output_dir=OUTPUT_FOLDER)
    pipeline.process_all_files()
