"""Step-level SWIFT: map each reasoning step to its token span inside a chat-formatted sequence.

The full text is rendered with the model's chat template first; each step's character range is recorded,
and the fast tokenizer's offset mapping assigns tokens to steps. This makes the mapping independent of the
model family (no hard-coded separator tokens). Steps are joined with "\\n\\n", following ProcessBench.
"""

SEP = "\n\n"


def build_example(problem, steps, tokenizer, system=None):
    """Return (input_ids, spans) where spans[k] = (start, end) token indices of step k (end exclusive)."""
    answer = SEP.join(steps)
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": problem},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    a0 = text.rfind(answer)
    assert a0 >= 0, "answer text not found in the rendered template"
    char_spans, pos = [], a0
    for s in steps:
        char_spans.append((pos, pos + len(s)))
        pos += len(s) + len(SEP)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    spans = []
    for cs, ce in char_spans:
        toks = [i for i, (a, b) in enumerate(offsets) if b > cs and a < ce and b > a]
        assert toks, "a step has no tokens"
        spans.append((toks[0], toks[-1] + 1))
    return ids, spans
