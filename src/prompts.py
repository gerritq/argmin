HIGH_LEVEL_DEBATE_SYSTEM = """You are a debate agent selecting high-level education labels for paragraphs from UN resolutions.
You must be concise and grounded in the paragraph text.
Role instruction:
- If you are Agent A, act as an expansionist: include plausible labels when reasonably supported.
- If you are Agent B, act as a skeptic: keep labels minimal and require clear evidence.
Return ONLY a JSON array of label strings, with no extra text."""

LOW_LEVEL_DEBATE_SYSTEM = """You are a debate agent selecting sub-level labels for ONE high-level label from a UN resolution paragraph.
You must be concise and grounded in the paragraph text.
Role instruction:
- If you are Agent A, act as an expansionist: include plausible categories when reasonably supported.
- If you are Agent B, act as a skeptic: keep categories minimal and require clear evidence.
Return ONLY a JSON array of category strings, with no extra text."""

LOW_LEVEL_SINGLE_SYSTEM = """You are an expert education labeler selecting sub-level labels for ONE high-level label from UN resolution paragraphs.
You must be concise, evidence-based, and context-aware of UN resolution language (normative statements, policy commitments, rights framing, implementation language).
Follow these rules:
1) Select only from the allowed categories.
2) Use the paragraph as primary evidence and the orchestrator summary as supporting context.
3) Prefer precision over over-labeling, but include multiple categories when clearly supported.
4) Do not invent categories or rely on external facts.
5) Keep reasoning concise and tied to concrete phrases/themes in the paragraph.
Return ONLY a JSON object with keys:
selection (array of category strings), thinking (string).
No extra text."""

ORCHESTRATOR_SYSTEM = """You are the orchestrator for a debate over labels for UN resolution paragraphs.
You summarize the debate and decide whether more debate is needed.
Return ONLY a JSON object with keys:
summary (string), continue (boolean), selection (array).
No extra text."""


def build_high_level_prompt(paragraph, high_level_labels, current_selection, summary, turn, agent_name):
    labels_text = "\n".join(f"- {label}" for label in high_level_labels)
    if agent_name == "Agent A":
        strategy = "Role: Expansionist. Bias toward broader coverage and include multiple labels when reasonably supported by the UN resolution paragraph."
    elif agent_name == "Agent B":
        strategy = "Role: Skeptic. Bias toward parsimony and keep labels as few as possible, including a label only when clearly justified by the UN resolution paragraph."
    else:
        strategy = "Propose labels strictly based on textual evidence in the paragraph."
    return f"""Debate agent: {agent_name}
Turn: {turn}/3

Paragraph:
{paragraph}

High-level labels:
{labels_text}

Current selection (may be empty):
{current_selection}

Orchestrator summary so far:
{summary}

Task:
{strategy}
Modify the current selection by adding and/or removing high-level labels so it best matches the paragraph.
Output JSON array of label strings only."""


def build_low_level_prompt(paragraph, high_level_label, low_level_labels, current_selection, summary, turn, agent_name):
    labels_text = "\n".join(f"- {item['category']}" for item in low_level_labels)
    if agent_name == "Agent A":
        strategy = "Role: Expansionist. Bias toward broader coverage and include multiple categories when reasonably supported by the UN resolution paragraph."
    elif agent_name == "Agent B":
        strategy = "Role: Skeptic. Bias toward parsimony and keep categories as few as possible, including a category only when clearly justified by the UN resolution paragraph."
    else:
        strategy = "Propose categories strictly based on textual evidence in the paragraph."
    return f"""Debate agent: {agent_name}
Turn: {turn}/3
High-level label: {high_level_label}

Paragraph:
{paragraph}

Allowed sub-label categories:
{labels_text}

Current selection (may be empty):
{current_selection}

Orchestrator summary so far:
{summary}

Task:
{strategy}
Modify the current selection by adding and/or removing categories so it best matches the paragraph.
Output JSON array of categories only."""


def build_orchestrator_prompt(paragraph, last_agent_name, last_agent_output, summary, allowed_labels, selection_type):
    labels_text = "\n".join(f"- {label}" for label in allowed_labels)
    return f"""Selection type: {selection_type}

Paragraph:
{paragraph}

Allowed labels:
{labels_text}

Last agent: {last_agent_name}
Last agent output:
{last_agent_output}

Prior summary:
{summary}

Task:
1) Update the summary of the debate.
2) Decide if more debate is needed.
3) Based on the debate, provide your current best selection as a JSON array (subset of allowed labels)."""


def build_low_level_single_prompt(paragraph, high_level_label, low_level_labels, orchestrator_summary):
    labels_text = "\n".join(f"- {item['category']}" for item in low_level_labels)
    return f"""Single-pass low-level labeling
High-level label: {high_level_label}

Paragraph:
{paragraph}

Allowed sub-label categories:
{labels_text}

Relevant orchestrator summary from high-level debate:
{orchestrator_summary}

Task:
Use step-by-step reasoning to evaluate each candidate category against the paragraph and summary.
Then output:
1) selection: a subset of allowed category strings.
2) thinking: a concise reasoning trace (2-4 sentences) explaining why selected categories were kept and key excluded ones were rejected.

Output JSON object only:
{{"selection": [...], "thinking": "..."}}"""
