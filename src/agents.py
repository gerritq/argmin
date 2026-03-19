import json
from typing import Any, Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import (
    HIGH_LEVEL_DEBATE_SYSTEM,
    LOW_LEVEL_DEBATE_SYSTEM,
    ORCHESTRATOR_SYSTEM,
    build_high_level_prompt,
    build_low_level_prompt,
    build_orchestrator_prompt,
)


class HFChatModel:
    def __init__(self, model_id: str):

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto"
        )

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 512, temperature: float = 0.2) -> str:
        encoded = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt"
        )
        input_ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(self.model.device)
        attention_mask = None
        if hasattr(encoded, "attention_mask") and encoded.attention_mask is not None:
            attention_mask = encoded.attention_mask.to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            attention_mask=attention_mask,
        )
        gen_ids = output_ids[0][input_ids.shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def _extract_json(text: str) -> Optional[Any]:
    starts = [text.find("{"), text.find("[")]
    starts = [s for s in starts if s != -1]
    if not starts:
        return None
    start = min(starts)
    for end in range(len(text), start, -1):
        snippet = text[start:end]
        try:
            return json.loads(snippet)
        except Exception:
            continue
    return None


def _normalize_selection(selection: Any, allowed: List[str]) -> List[str]:
    if not isinstance(selection, list):
        return []
    allowed_set = set(allowed)
    return [item for item in selection if isinstance(item, str) and item in allowed_set]


class DebateAgent:
    def __init__(self, name: str, system_prompt: str, model: HFChatModel):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    def propose_high_level(
        self,
        paragraph: str,
        high_level_labels: List[str],
        current_selection: List[str],
        summary: str,
        turn: int,
    ) -> List[str]:
        prompt = build_high_level_prompt(
            paragraph, high_level_labels, current_selection, summary, turn, self.name
        )
        text = self.model.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _extract_json(text)
        return _normalize_selection(parsed, high_level_labels)

    def propose_low_level(
        self,
        paragraph: str,
        high_level_label: str,
        low_level_labels: List[Dict[str, str]],
        current_selection: List[str],
        summary: str,
        turn: int,
    ) -> List[str]:
        allowed_codes = [item["code"] for item in low_level_labels]
        prompt = build_low_level_prompt(
            paragraph, high_level_label, low_level_labels, current_selection, summary, turn, self.name
        )
        text = self.model.chat(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _extract_json(text)
        return _normalize_selection(parsed, allowed_codes)


class Orchestrator:
    def __init__(self, model: HFChatModel):
        self.model = model

    def summarize_and_judge(
        self,
        paragraph: str,
        last_agent_name: str,
        last_agent_output: List[str],
        summary: str,
        allowed_labels: List[str],
        selection_type: str,
    ) -> Dict[str, Any]:
        prompt = build_orchestrator_prompt(
            paragraph,
            last_agent_name,
            json.dumps(last_agent_output, ensure_ascii=True),
            summary,
            allowed_labels,
            selection_type,
        )
        text = self.model.chat(
            [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = _extract_json(text)
        if not isinstance(parsed, dict):
            return {"summary": summary, "continue": True, "selection": []}
        selection = _normalize_selection(parsed.get("selection"), allowed_labels)
        return {
            "summary": str(parsed.get("summary", summary)),
            "continue": bool(parsed.get("continue", True)),
            "selection": selection,
        }


class AgenticLabeler:
    def __init__(self, model_id: str):
        self.model = HFChatModel(model_id)
        self.debate_a = DebateAgent("Agent A", HIGH_LEVEL_DEBATE_SYSTEM, self.model)
        self.debate_b = DebateAgent("Agent B", HIGH_LEVEL_DEBATE_SYSTEM, self.model)
        self.low_a = DebateAgent("Agent A", LOW_LEVEL_DEBATE_SYSTEM, self.model)
        self.low_b = DebateAgent("Agent B", LOW_LEVEL_DEBATE_SYSTEM, self.model)
        self.orchestrator = Orchestrator(self.model)

    def _run_high_level_debate(self, paragraph: str, high_level_labels: List[str]) -> List[str]:
        summary = ""
        current_selection: List[str] = []

        print("=" * 60)
        print(f"BEGIN HIGH-LEVEL DEBATE: {paragraph[:60]}...")
        print("=" * 60)
        for turn in range(1, 4):
            for agent in (self.debate_a, self.debate_b):
                print(f"\nTURN {turn} AGENT {agent.name} SUMMARY:\n{summary}\n ")
                proposal = agent.propose_high_level(
                    paragraph, high_level_labels, current_selection, summary, turn
                )
                print(f"PROPOSAL: {proposal}")
                result = self.orchestrator.summarize_and_judge(
                    paragraph,
                    agent.name,
                    proposal,
                    summary,
                    high_level_labels,
                    "high_level",
                )
                print(f"ORCHESTRATOR RESULT: {result}")
                summary = result["summary"]
                current_selection = result["selection"] or proposal or current_selection
                if not result["continue"]:
                    return current_selection
        return current_selection

    def _run_low_level_debate(
        self, paragraph: str, high_level_label: str, low_level_labels: List[Dict[str, str]]
    ) -> List[str]:
        summary = ""
        current_selection: List[str] = []
        allowed_codes = [item["code"] for item in low_level_labels]

        print("=" * 60)
        print(f"BEGIN LOW-LEVEL DEBATE: {paragraph[:60]}...")
        print("=" * 60)
        for turn in range(1, 4):
            for agent in (self.low_a, self.low_b):
                print(f"\nTURN {turn} AGENT {agent.name} SUMMARY:\n{summary}\n ")
                proposal = agent.propose_low_level(
                    paragraph, high_level_label, low_level_labels, current_selection, summary, turn
                )
                print(f"PROPOSAL: {proposal}")
                result = self.orchestrator.summarize_and_judge(
                    paragraph,
                    agent.name,
                    proposal,
                    summary,
                    allowed_codes,
                    "low_level",
                )
                summary = result["summary"]
                current_selection = result["selection"] or proposal or current_selection
                if not result["continue"]:
                    return current_selection
        return current_selection

    def label_paragraph(self, paragraph: str, label_hierarchy: Dict[str, List[Dict[str, str]]]) -> Dict[str, Any]:
        high_level_labels = list(label_hierarchy.keys())
        selected_high = self._run_high_level_debate(paragraph, high_level_labels)

        low_level: Dict[str, List[Dict[str, str]]] = {}
        # for high_label in selected_high:
        #     low_labels = label_hierarchy.get(high_label, [])
        #     selected_codes = self._run_low_level_debate(paragraph, high_label, low_labels)
        #     selected_items = [item for item in low_labels if item["code"] in set(selected_codes)]
        #     low_level[high_label] = selected_items

        return {
            "high_level": selected_high,
            "low_level": low_level,
        }
