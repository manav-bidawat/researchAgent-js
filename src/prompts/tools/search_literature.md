Search arXiv for papers on a topic, download them, and add them to the index. Use this
when the corpus has no papers on the subject being asked about, or when
`retrieve_evidence` reported insufficient evidence and the gap looks like missing
literature rather than a bad query. This is slow (30–60s) and costs API calls — do not
use it to answer questions about papers that are already indexed; use
`retrieve_evidence` for that. Do not call it twice for the same topic.
