# Babbling Baby

A tabula-rasa byte-level transformer learning to babble by playing a
turn-taking word-emission game against a dictionary-sampling teacher,
with a WordNet-grounded verifier and an adaptive length-based
curriculum.

The whole experiment runs on a single laptop. No pretraining, no
cloud, no curated corpus, no human labelling. The pupil starts with
random weights and reaches the full strict English dictionary
(96 715 words) as its working vocabulary after an overnight run.

The spec is at [`babbling-baby.draft.md`](babbling-baby.draft.md) in
[DraftLang](https://github.com/elepedus/draft-lang) format.

---

## Why this exists

This is v0 of a broader architectural picture for tabula-rasa LLM
bootstrapping through observable teacher behaviour and grounded
verifiers — a pupil that learns to anticipate intelligent agents
rather than absorbing a static corpus, the same way AlphaZero learns
to play games through self-play with a win/loss verifier.

That broader picture has many open questions (multi-teacher
composition, environment grounding, peer challenge, evolutionary
architecture search). Babbling Baby tests the single load-bearing
assumption underneath all of them: **can a small, randomly initialised
model on commodity hardware learn structurally meaningful language
output through behavioural prediction plus a grounded verifier?**

If this hypothesis fails, the rest of the architecture collapses. If
it holds — even at the babbling level — the rest gets the right to be
investigated. So v0 is the cheapest possible probe of the load-bearing
claim, not the architecture itself.

## The hypothesis under test

> Given a small, randomly initialised byte-level transformer, a
> queryable teacher emitting dictionary words, and a verifier that
> scores emissions against a real-word criterion, the pupil's
> dictionary-word emission rate can be driven meaningfully above its
> untrained baseline on a single M2 Max in tractable time.

Babbling, not understanding. Sounds shaped like a language, not
sentences that mean anything.

## The system

| Piece | What it does |
|---|---|
| `Pupil` | Byte-level decoder-only transformer (~1.6 M params, 4 layers, 256 dim, 4 heads, 128 context). Random init. |
| `DictionaryTeacher` | Samples a word from the current curriculum level. Fast, no API calls. |
| `Verifier` | `is_word(emission)`: length ≥ 3 AND has a WordNet synset. Excludes letter-combos like `se`/`ss`/`st`/`sm`/`sta`. |
| `Curriculum` | Adaptive length-based; six levels (1-letter → 2-letter → 3-letter → 4-5 → 6-8 → full strict). Per-level unlock thresholds 0.85 → 0.10. Active mixture is focus ± 1. |
| `Game loop` | Each turn: teacher emits a word + space; pupil trains imitation on it; pupil samples its own word; verifier scores it; RL gradient applied if it scored. |

The space delimiter matters: without it the pupil emits one
unbroken letter stream. The verifier strictness matters: it forces
"real word" to mean something a native speaker would call a word,
not a chemical symbol or compass abbreviation. The curriculum's
overlapping levels matter: foundational mastery is maintained while
the next rung gets early exposure.

## Running it

```bash
uv sync --extra dev          # one-time
uv run pytest                # 57 tests, ~10 seconds (skip test_acceptance if in a hurry)
uv run python scripts/run_training.py 600 5      # 10-minute run, log every 5s
uv run python scripts/plot_trace.py runs/<id>/trace.jsonl
```

Long runs:

```bash
uv run python scripts/run_training.py 28800 60   # 8 hours, log every 60s
```

Each run writes to `runs/<timestamp>/`:

- `trace.jsonl` — per-interval metrics: loss, four trace metrics, focus level, per-level success rates, sample preview
- `levels.jsonl` — focus-level transitions with timestamps
- `pupil.npz` — final saved weights (~13 MB)
- `trace.png` — four-panel metric plot plus focus track and per-level rates

## What the overnight run showed

8 hours of training on an M2 Max MacBook (96 GB unified memory), one
process, no GPU outside the M2's. Pupil starts at random init.

### Focus progression

| transition | wall-clock | gap |
|---|---|---|
| → L1 (2-letter) | 1.7 s | 1.7 s |
| → L2 (3-letter) | 26.5 s | 25 s |
| → L3 (4–5 letter) | 3 m 5 s | 2 m 38 s |
| → L4 (6–8 letter) | 2 h 24 m | 2 h 22 m |
| → L5 (full strict) | 5 h 18 m | 2 h 54 m |
| end (8 h) | — | 2 h 41 m at L5 |

The first three levels fall in three minutes. Each subsequent
transition costs ~2.5 hours of training. The wall isn't capacity —
it's the time it takes to reach mastery at each new vocabulary tier.

### Metrics over time

| hour | focus | loss EMA | %letters | %shape | %dict |
|---|---|---|---|---|---|
| 0 | L2 | 2.43 | 0.71 | 0.35 | 0.09 |
| 1 | L3 | 2.33 | 0.66 | 0.24 | 0.06 |
| 3 | L4 | 2.28 | 0.77 | 0.48 | 0.10 |
| 5 | L4 | 2.16 | 0.84 | 0.58 | 0.12 |
| 6 | L5 | 2.03 | 0.79 | 0.56 | 0.17 |
| 8 | L5 | 1.97 | 0.78 | 0.60 | 0.21 |

`%dict` (strict, WordNet-grounded): 0.04 baseline → **0.21 at end**.
One in five chunks at the end of the run is a real dictionary word.

### Final sample at L5

```
shepaws sinds ppos arn f chiatalel ctics meine p ps mes rices
rinalinon ricasa try phoca gona lonary phetin on ches ehes lininin
```

Picked-out real words: `arn`, `f`, `mes`, `rices`, `try`, `phoca`
(genus of seals), `gona`, `on`. The non-word fragments — `shepaws`,
`chiatalel`, `meine`, `rinalinon`, `phetin`, `lininin` — have
distinctly English prosody. Not English, but unmistakably
English-shaped.

402 unique strict-real words across the full run. From the final
hour alone: `bones`, `bruchus`, `caste`, `cive`, `dine`, `jar`,
`line`, `lode`, `logged`, `naves`, `phoca`, `phonies`, `ring`,
`try`, `who`.

## What this validates

From the spec's claim inventory, ordered by how cleanly the data
moves them:

- **Prediction signal alone, with a grounded verifier, can drive a
  tabula-rasa byte-level transformer to produce structurally
  meaningful language output on commodity hardware.** Cleanly
  supported. This is the v0 hypothesis and it held up.
- **The byte-level substrate works.** The pupil learned letter
  patterns, word boundaries, multi-character word structure, and
  vocabulary, all from raw bytes. No tokenisation tricks required.
- **An adaptive curriculum is the right shape of pedagogy.** The
  model graduated through six discrete competence tiers under its
  own success signal. Each tier reflected real capability, not just
  threshold-crossing luck.
- **Embodiment (in the digital sense) wasn't necessary for v0.** The
  pupil never touched a VM. A dictionary verifier was enough.

## What this doesn't validate

The broader architectural picture remains untested:

- Multi-teacher behavioural prediction (we used one).
- Architecture-independent protocol (we used one architecture).
- Peer-network mutual challenge (no peers).
- Evolutionary architecture search (architecture was fixed).
- Linux VM as embodiment (digital world, but not a rich one).
- Anticipation as a distinct mechanism from mimicry (we just used
  imitation + verifier RL).

The path to those tests is wide open; this is the first step on it.

## Lessons from the iteration

A few diagnostic moments worth recording, since they're easy to
forget and the spec doesn't capture them:

- **Loose verifier credits were largely from possessive apostrophes**
  acting as accidental word boundaries. Tightening the verifier
  surfaced that the model had no real word boundaries; the apostrophes
  were doing the work. Fix: explicit space delimiter in the game
  protocol.
- **Length-shaped penalty signals were destructive.** Strong negative
  rewards on short emissions made the model retreat from emitting
  anything at all. Fix: positive-only rewards (+1 for hits, 0 for
  misses); let the curriculum supply the difficulty gradient.
- **RL gradients touching the trailing space were fighting the
  imitation signal.** The imitation step trains the space as the
  right next byte; RL was sometimes pushing against it. Fix: strip
  the delimiter before applying RL.
- **Discrete-jump promotion blew through low levels too fast.**
  Per-level thresholds (high for easy, low for hard) gave deeper
  mastery and more sustained progression.

## What v1 should test

In rough order of how much they push on the broader picture:

1. **Batched and population training**: a benchmark
   (`scripts/bench_batch.py`) shows the GPU runs at ~1–2 %
   utilisation at batch size 1. At batch 64 the M2 Max processes 41×
   more examples per second for ~1.5× more wall-clock per step. The
   overnight run could plausibly become a ~15-minute run at batch 64.
   And population training is real headroom — stack N pupils into a
   batched super-tensor, vmap the forward, train a population in one
   pass. This is the cheapest first move on any subsequent
   experiment.
2. **Bigger pupil + longer training**: does the L5 rate climb past
   21 % toward something resembling real fluency? At what model size
   does each level transition stop costing hours? (Combined with #1,
   "longer training" is much cheaper than 8-hour wall-clock would
   suggest.)
3. **Diffusion–AR hybrid architecture (TiDAR)**: adapt the byte-level
   pupil to Nvidia's TiDAR recipe (Liu et al. 2025) — same model
   does parallel diffusion drafting and AR-quality sampling in one
   forward pass via structured attention masks, with 4.7–5.9×
   throughput speedup at AR quality parity. The byte-level +
   diffusion combination is also the natural substrate for image
   generation (raw pixel bytes as parallel-output targets), making
   this a foundation for the multimodality work below.
4. **Phrase-level curriculum**: same mechanism, longer units. Extend
   the curriculum from single words to grammatical phrases of
   increasing complexity (`Cat sat.` → `Three blind mice.` →
   `The cat sat on the mat.`), with a grammaticality verifier
   replacing the dictionary lookup. Tests whether
   prediction-plus-verifier extends from lexical into syntactic
   structure.
5. **Multimodality and cross-modal equivalence**: train the same
   byte-level architecture on text, audio (PCM or codec bytes), and
   image bytes (raw pixels or post-decode PNG), then on mixed
   streams. The strongest version: can the model learn that audio
   of "cat" ≡ text `cat` ≡ image of a cat, purely from prediction
   over their interleaved byte streams? Same prediction objective,
   no modality-specific loss terms. Especially interesting paired
   with the TiDAR variant above.
6. **Multi-teacher**: add Claude or a local LLM as a richer teacher
   alongside the dictionary sampler. Does the pupil acquire structure
   the dictionary alone can't teach?
7. **Anticipation objective**: train on predicting the teacher's
   *next emission given trace prefix* rather than just imitation. Does
   that produce qualitatively different learning?
8. **Peer challenge**: instantiate two pupils, have them play each
   other's hard problems. Does the population mechanism work at this
   scale? (Cheap via population training in #1.)
9. **VM embodiment**: bolt the pupil onto a Linux container with
   tool-use over real commands. Does grounding in actual
   action/consequence shift what gets learned?

The intent isn't to do all of these. It's to pick the one that most
sharply probes the next open claim in the inventory.

## Project layout

```
babbling-baby/
├── babbling-baby.draft.md     # DraftLang spec — what the system promises
├── README.md                  # this file
├── pyproject.toml             # mlx, anthropic, pyspellchecker,
│                              #   wordfreq, nltk, matplotlib, pytest
├── src/babbling_baby/
│   ├── verifier.py            # is_word, random_word (WordNet-grounded)
│   ├── teacher.py             # DictionaryTeacher (curriculum-driven)
│   ├── pupil.py               # byte-level decoder transformer + sampling
│   ├── curriculum.py          # adaptive graded curriculum
│   ├── trace.py               # four-granularity progress metrics
│   └── train.py               # interactive imitation + RL training loop
├── tests/                     # 57 tests; outside-in BDD discovered them
├── scripts/
│   ├── run_training.py        # entry point — duration + log interval
│   └── plot_trace.py          # six-panel diagnostic plot
└── runs/                      # output of each training session
    └── <timestamp>/
        ├── trace.jsonl
        ├── levels.jsonl
        ├── pupil.npz
        └── trace.png
```

## Acknowledgements

The architectural picture this experiment lives inside emerged from a
long conversation about whether AlphaZero-style tabula-rasa
bootstrapping could be adapted for language models. The Doctor Who
*Midnight* metaphor for behavioural prediction, the AlphaZero
self-play loop, and the adaptive-curriculum framing all came from
that conversation; this codebase is the smallest concrete artefact
the framing implies.
