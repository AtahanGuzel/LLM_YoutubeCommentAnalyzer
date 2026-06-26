EVAL1_PROMPT = """You are auditing the label quality of a comment extraction model.

For each comment below, determine whether the assigned label is correct.
You must return a verdict for EVERY comment — do not skip any.

COMMENTS WITH ASSIGNED LABELS:
{sampled_comments}

Output a JSON array of exactly {n} verdict objects. Each object must contain:
- comment_id: the comment identifier
- correct: true if the label is correct, false if it contains an error
- issue: null if correct is true; a brief description of the error if correct is false

Output JSON only:
[
  {{"comment_id": "001", "correct": true, "issue": null}},
  {{"comment_id": "012", "correct": false, "issue": "competitor Sony missed; comment says 'switched from Sony' but competitor_mentions is empty"}}
]"""
