EVAL2_PROMPT = """You are evaluating the output quality of a comment analysis model.
Present evidence before stating any score. Do not state scores until
you have completed the evidence review for each criterion.

TASK: Analyze YouTube comments about {primary_product} for sentiment,
      pain points, and competitor mentions.

SAMPLED COMMENTS WITH SCOUT 17B LABELS:
{sampled_comments}

SCOUT 17B FINAL AGGREGATED OUTPUT:
{aggregated_output}

Output JSON only in the structure below.

{
  "output_quality": {
    "sentiment": {
      "evidence": "<specific observations from samples>",
      "gaps": "<contradictions or missed patterns, or null>",
      "score": <1-5>
    },
    "pain_points": {
      "evidence": "<specific observations from samples>",
      "gaps": "<contradictions or missed patterns, or null>",
      "score": <1-5>
    },
    "competitors": {
      "evidence": "<specific observations from samples>",
      "gaps": "<contradictions or missed patterns, or null>",
      "score": <1-5>
    }
  }
}

Anchored scoring rubrics:

Criterion 1 — Sentiment classification accuracy:
5 → Assigned sentiment is clearly supported by the majority of sampled comments.
    No meaningful contradicting evidence present.
4 → Well supported. A small number of samples suggest mild ambiguity but
    don't contradict the overall classification.
3 → Partially supported. Roughly equal evidence exists for an alternative
    classification.
2 → Weakly supported. Stronger evidence in samples points toward a different
    classification.
1 → Directly contradicts the majority of sampled comments.

Criterion 2 — Pain point identification accuracy:
5 → All identified pain points are directly evidenced in samples.
    No recurring problem pattern in samples was missed.
4 → Well evidenced. One minor issue is slightly mischaracterized or a
    low-signal pattern was missed.
3 → Most pain points valid but one is not well supported, or one clearly
    recurring issue is absent.
2 → Multiple pain points weakly supported or a high-frequency problem
    visible in samples does not appear in output.
1 → Identified pain points do not reflect sampled comment content.

Criterion 3 — Competitor identification accuracy:
5 → All competitors in sampled comments correctly captured.
    Threshold rule correctly applied.
4 → Correctly identified with minor characterization variance.
    No missed competitor from samples.
3 → One competitor visible in samples absent from output, or one included
    competitor weakly evidenced.
2 → Multiple competitors missed or incorrectly included relative to samples.
1 → Competitor output does not reflect sampled comment content."""
