Retrieve passages, figures, and tables from the indexed papers that are relevant to a
query. This is your primary source of evidence — use it for any question about what the
papers say. Returns ranked chunks with paper, page, and section metadata so you can cite
them precisely. If `sufficient_evidence` comes back false, the index does not contain
good evidence for this query: either rephrase and retry with different wording, or — if
the topic is genuinely absent from the corpus — call `search_literature`, or tell the
user you cannot answer from available evidence. Do not answer from your own knowledge
when this returns nothing.
