HIGH_LEVEL_DEBATE_SYSTEM = """You are a debate agent selecting high-level education labels.
You must be concise and grounded in the paragraph.
Return ONLY a JSON array of label strings, with no extra text."""

LOW_LEVEL_DEBATE_SYSTEM = """You are a debate agent selecting sub-level labels for ONE high-level label.
You must be concise and grounded in the paragraph.
Return ONLY a JSON array of category strings, with no extra text."""

ORCHESTRATOR_SYSTEM = """You are the orchestrator.
You summarize the debate and decide whether more debate is needed.
Return ONLY a JSON object with keys:
summary (string), continue (boolean), selection (array).
No extra text."""


def build_high_level_prompt(paragraph, high_level_labels, current_selection, summary, turn, agent_name):
    labels_text = "\n".join(f"- {label}" for label in high_level_labels)
    if agent_name == "Agent A":
        strategy = "Bias toward broader coverage: propose multiple labels when they are reasonably supported by the paragraph."
    elif agent_name == "Agent B":
        strategy = "Bias toward parsimony: keep the proposed labels as few as possible, including a label only when clearly justified."
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
        strategy = "Bias toward broader coverage: propose multiple categories when they are reasonably supported by the paragraph."
    elif agent_name == "Agent B":
        strategy = "Bias toward parsimony: keep the proposed categories as few as possible, including a category only when clearly justified."
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
