PREPROCESSING_PROMPT = """Think step by step:
1. What is the central topic of this video?
2. Is a specific physical product the main subject?
3. Is the video's purpose to review, showcase, compare, or promote that product?
4. If yes, what is the exact product name?
5. If multiple products are present, is one clearly the primary subject?

If your answer to step 3 is yes for any qualifying purpose — review, showcase,
compare, or promote — set is_promotion to true. A third-party review counts.

Output JSON only:
{
  "is_promotion": true | false,
  "reasoning": "one sentence",
  "primary_product": "exact product name or null",
  "product_confidence": "high" | "low",
  "candidates": ["product1", "product2"] | null
}"""
