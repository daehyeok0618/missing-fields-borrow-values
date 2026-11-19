# Experiment 2: prospective founding-year generalization

Client-date plan: 2026-11-02. Research record, not a student-authored final manuscript.
The user authorized the recommended second-attribute replication. This experiment
is informed by the completed Salary study, whose frozen results are preserved in
reference/. No new-year outputs have been examined when this protocol is locked.

## Research question and decisions

Does Qwen's attribute-dependent donor selection and placement sensitivity also occur
when the missing target field is a founding year rather than salary?

Keep the same model revision, FP16, SDPA, greedy decoding, repetition penalty 1.1,
max 96 generated tokens, batch size 16, JSON string/null schema, and strict parser.
Use newly generated organizations and year triples; no Salary inference is repeated.

Strict primary H1: delta_attr = [(p(v1|S1)-p(v1|S2)) +
(p(v2|S2)-p(v2|S1))]/2 > 0, averaged equally over templates/orders/placements.
Strict primary H2 requires BOTH delta_C=C_adjacent-C_middle>0 AND
delta_type=delta_C-(U_adjacent-U_middle)>0. C matches either donor candidate;
U is a valid string absent from the complete input messages after normalization.
Each required effect must have its 95% percentile CI entirely above zero. A CI
including zero leaves that prediction unestablished; a CI entirely below zero
provides evidence against its direction. We do not redefine hypotheses after results.

## Independent new data

96 main bases and 12 development bases. EACH base has all three templates (KV,
prose, bullet), two donor orders and two placements. Names are disjoint from prior
local donor-study base files (including the completed Salary study) and across
development/main. Each base has three distinct four-digit years drawn uniformly
without replacement from 1900–2025. Unordered year triples do not repeat across
bases/splits, but individual years may repeat. Candidate v1 is smaller than v2 in
half the bases. Target gold year differs from both donor years.

S1: donor 1 founding year=v1, donor 2 audit year=v2.
S2: donor 1 audit year=v1, donor 2 founding year=v2.
N0: donor 1 audit year=v1, donor 2 review year=v2.
Target founding year is omitted in MISSING and N0; no missingness sentence is added.
PRESENT explicitly adds a third, supported target founding year. T0/T1 contain
only the target document and neutral text.

Per base/template: MISSING 2 assignments x 2 orders x 2 placements=8 calls;
PRESENT likewise 8; N0 2 orders x 2 placements=4; T0/T1=2. Thus 66 calls/base,
6,336 main calls (2,304 primary) and 792 development calls.

Target first, final question last. Middle: target / neutral prefix / contiguous
donors / neutral suffix / question. Adjacent: target / same prefix / same suffix /
same donors / question. Main total lengths 2,000 +/-64 chat-templated tokens.
Placement pairs must have identical lengths, text multisets and query token spans.
Middle donor-block midpoint must be 45–55% of input; Adjacent has no intervening
content. S1/S2 pairs must preserve numeric token spans and total lengths. The KV
field prefix `Recorded founding year` / `Recorded audit year` uses equal token
lengths; field wording is chosen by tokenizer audit before model inference.

Reuse the exact archived public-domain Gutenberg ebook 11 downloaded for Salary.
Preserve source/hash/download provenance. Filter out paragraphs containing digits
or whole-word founding/audit/review/salary/budget/revenue/USD, and verify absence of
example names/values. Use original paragraphs or word-boundary prefixes, no synthetic
filler. Each template has its own length-matched excerpts, as in Salary. Therefore
template differences are not pure wording effects. This study does not independently
vary corpus or model, and no internal representation interventions are performed.

## Strict primary versus prespecified semantic secondary

Strict parser: exactly one JSON object with only `value`, either JSON null or a
nonempty string; no fences, duplicate keys, bare values, numeric JSON values,
booleans or automatic repairs. Normalize NFKC, lowercase, remove punctuation/symbols,
collapse whitespace. Match exact donor values before checking whole-value occurrence
in all message contents. Invalid outputs stay in every denominator with zero selection
indicators. Supported target answers are scored separately before error taxonomy.

The Salary study contained many quoted `"null"` values. This time the SECONDARY
semantic policy is fixed BEFORE inference: normalize and map exactly
{null, na, none, unknown, not specified, not provided, not available, not stated,
unspecified} from valid string values to abstention. `n/a` normalizes to `na`.
No other strings or invalid JSON are repaired. Report literal null, quoted null,
other aliases and non-abstention values separately, plus strict and semantic H1/H2.
The strict result is the registered primary; consistency under semantic treatment
is a separate robustness statement. Do not select whichever policy is significant.
Numeric versus nonnumeric context-absent strings are reported separately.

## Development gate and execution discipline

Freeze code/config/source/data/statistics before development inference. Require:
- all expected development requests completed without system error/truncation;
- >=95% strict schema-valid across development;
- T0 target-only missing semantic accuracy >=90%;
- T1 target-only present strict exact accuracy >=90%.

PRESENT multi-document accuracy is diagnostic, not an exclusion criterion. Donor
effect size or hypothesis significance is NEVER a gate. If baseline capability fails,
preserve diagnostics and stop main rather than automatically switching attributes,
tuning prompts or cherry-picking examples. No unplanned semantic repair is allowed.
Pilot throughput extrapolates main cost; target <=2 hours/run, split execution if
needed. Hardware is one local RTX A6000, not the HW1-planned Colab T4. Save actual
runtime/VRAM/environment. System interruptions resume fixed batch memberships;
do not regenerate bad answers. Sampling is disabled; seed is not an independent run.

## Statistics and comparison with Salary

Same paired base-cluster bootstrap: 2,000 resamples, fixed seed 2026110203, retaining
every rendering/condition of a base. Use one index matrix for all year contrasts,
including semantic sensitivity. Report effect CIs, counts/outcome rates, both full
switch directions, per-template/order and leave-one-template-out results. Secondary
CIs are pointwise, not simultaneous. Flag sparse context-absent events and explain
that zero-event [0,0] bootstrap intervals are not zero population-risk proofs.

Tabulate frozen Salary strict H1/H2 beside Year strict H1/H2. Also show the two
fields under the SAME null/N/A secondary policy, using the saved Salary sensitivity
and a matching Year sensitivity. The broader Year alias policy is separately named.
Datasets, labels, value lengths, query wording and some context excerpts differ:
do not call their effect-size difference the causal effect of attribute identity,
and do not pretend cases are paired across tasks. No new cross-task p-value is planned.

Outcomes may support both hypotheses, only one, or neither. Complete generation,
analysis, raw-data audit, plots, Korean research report and a reproducibility ZIP.
Report exactly which hypotheses generalize, control quality, alias sensitivity,
template variation and remaining model/corpus/length limitations. Preserve all
prior Salary artifacts and their hashes.
