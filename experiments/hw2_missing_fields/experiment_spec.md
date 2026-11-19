# HW2 experiment 1: submitted HW1 label-swap and matched-placement study

This is an execution protocol, not submission-ready student prose. The submitted
`22_When_Missing_Fields_Borrow_.pdf`, sections 4–5, is the design reference.
Previous experiments are exploratory context; none of their outputs enters this study.

## Development-only representation amendment (v2, before any main generation)

The initial v1 development run completed 792 outputs with valid JSON throughout.
T0 was 36/36, while T1 was 0/36 under the fixed normalization because the model
copied the correct amount with a lexical USD prefix (for example `USD 194957`).
USD is a word, not punctuation, so the specified lowercasing/punctuation/whitespace
normalization cannot equate it with the bare candidate amount. This made the
control diagnose unit rendering instead of amount retrieval. The v1 source,
protocol lock, data and outputs are preserved in audit/protocol_v1_before_currency_format_fix.zip
and the original runs/run_20261031_hw1_main directory.

V2 changes only document currency representation from `USD 194957` to `$194957`
(for every target and donor amount). The existing symbol normalization already
handles dollar signs. No parser rule, hypothesis, candidate amount, entity identity,
template family, generation setting, sample size or statistical decision rule is
changed. Padding is regenerated to restore matched lengths. The same dev cases
are reused as development; main has not been generated or inspected. This is a
documented format refinement based on development controls, not an untouched
pre-pilot protocol. V2 is separately hash-locked before rerunning development and
then main. A repeated control failure will be reported, not repaired indefinitely.

## Hypotheses and primary population

H1: delta_attr = 0.5 * [(p(v1|S1)-p(v1|S2)) + (p(v2|S2)-p(v2|S1))] > 0.
Pool equally over three renderings of EACH base, both donor orders, and both placements.
H2: delta_C = C_adjacent-C_middle > 0 AND
delta_type = delta_C-(U_adjacent-U_middle) > 0, where C matches EITHER donor
candidate and U is a valid output absent from the complete input message contents.
H1 is supported only if its 95% percentile CI excludes zero on the positive side.
H2 is supported only if BOTH required CIs do so. An interval spanning zero leaves
the relevant effect unestablished; an entirely negative interval contradicts its direction.

## Data and conditions

96 new independent base examples; each base is rendered in KV, prose, and bullet
templates with the same entities and amounts. Missing target salary is omitted,
not replaced by an explicit missingness sentence. Two distinct donor documents
hold two distinct six-digit amounts, each occurring exactly once. Salary/Budget
are both compatible with monetary values. Names and all amounts are new relative
to locally stored previous donor-study base data and disjoint across dev/main.

For each base/template:
- MISSING: S1/S2 x AB/BA x MIDDLE/ADJACENT = 8 calls.
- N0: Budget/Revenue x AB/BA x MIDDLE/ADJACENT = 4 calls.
- PRESENT: supported target value with S1/S2 x orders x placements = 8 calls.
- T0/T1: target-only missing/present, also padded near 2,000 tokens = 2 calls.

Total: 96 x 3 x 22 = 6,336 calls; primary missing factorial = 2,304.
Development: independent 12 x 66 = 792 calls, never included in inference statistics.
Target is always the FIRST document. Question is always last. Two donors are contiguous.
Middle = target, fixed neutral prefix, donor block, neutral suffix, question.
Adjacent = target, same prefix, same suffix, same donor block, same question.
Both layouts contain exactly the same text blocks. Middle donor-block midpoint
must lie at 45–55% of the full chat-templated input length. Adjacent has no content
block between donor block and question. Placement pairs have EXACTLY equal token
counts; S1/S2 pairs also have equal token counts and fixed numeric token spans.
Lengths must be 2,000 +/- 64 tokens. Actual spans/lengths are retained for every request.

Neutral text: filtered excerpts of Project Gutenberg ebook 11, Alice's Adventures
in Wonderland (English fiction unrelated to organization salary extraction).
The complete downloaded public text and its URL/hash are saved. Paragraphs are
whitespace-normalized; a final paragraph may be cut at a word boundary to approach
the token budget. No generated filler is used. Exclude digit-containing paragraphs,
all manipulated attribute labels, USD, and each example's entity names and values.
Paragraph identities and selected excerpts are retained. Findings are conditional
on this corpus and fixed prompt templates, not all neutral text or templates.

## Model and output evaluation

Qwen2.5-1.5B-Instruct fixed revision, official chat template, FP16, greedy,
released repetition penalty 1.1 explicitly fixed, max 96 new tokens; no grammar,
candidate enumeration, fine-tuning, or teacher-forced scoring. Local A6000 replaces
the planned Colab T4; this infrastructure difference is disclosed with measured runtime.

Strict parser accepts only a JSON object with exactly one `value` key and either
a nonempty string or JSON null. Duplicate keys, code fences, extra text, numbers
instead of strings, arrays, booleans, and empty/punctuation-only strings are invalid.
No automatic semantic repair. Normalization = Unicode NFKC, lowercase, deletion
of punctuation/symbols, and collapsed whitespace. A string such as "null" is NOT
an abstention. For in-context matching, require normalized whole-value boundaries
in the complete system/user message contents, after checking exact donor matches.

Missing/N0 taxonomy: ABSTAIN, DONOR_MATCH, OTHER_IN_CONTEXT, CONTEXT_ABSENT, INVALID.
Also retain queried-attribute versus competing-attribute donor identity. For present
controls, target correctness is scored FIRST and reported separately; a correct
target amount must never be counted as an unsupported output. Runtime failures
stop and are logged; no request is silently dropped or regenerated for its answer.
All invalid outputs remain in denominators and have zero donor/context-absent indicators.
Truncation is recorded, with the raw text parsed by the same strict rule.

## Statistics, checks, and decision discipline

2,000 base-cluster bootstrap resamples, fixed common index matrix for every contrast.
All templates, assignments, orders, placements and controls for a resampled base
travel together. CIs condition on these three templates and this fixed protocol.
Report category counts/rates, H1/H2 contrasts, both directions of complete value
switches, per-template/order effects and leave-one-template-out estimates.
All subgroup/LOTO findings are secondary pointwise CIs, not new primary hypotheses.
Zero-event bootstrap intervals do not establish zero population risk; include
zero-event base-level upper bounds where appropriate and flag sparse U counts.

Before main: static dataset/parser/contrast checks, source/data/code/config hash lock,
then development run. Pilot checks technical completion, output parsing, basic
controls, memory and extrapolated cost. Low behavioral effects or weak control
accuracy are reported but do NOT trigger prompt changes, example selection, or
cancellation of the registered hypotheses. Technical failures must be repaired
and logged before proceeding. The semantic protocol will not be tuned on pilot
effects. Split runs if projected duration exceeds two hours per run.

Preserve prompts, gold labels, token spans, generated token IDs, raw outputs,
model/tokenizer revisions, runtime, bootstrap indices, request/base-level results,
figures, reproducibility commands, AI usage and a Korean research record.
