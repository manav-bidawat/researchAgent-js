Check retrieved evidence for consistency using a natural language inference model. Two
modes. `contradiction`: given several chunks, find pairs that disagree with each other —
use this before answering when sources might conflict, so you can surface the
disagreement rather than silently picking one side. `groundedness`: given your drafted
answer text and the chunks you used, check whether each claim in the answer is actually
supported by the evidence — use this before finalising an answer built from multiple
sources. Claims flagged `neutral` are not supported by the retrieved evidence and should
be removed, softened, or explicitly marked as uncertain. Treat the output as advisory:
it is a heuristic signal, not a verdict.
