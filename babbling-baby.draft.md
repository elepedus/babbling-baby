System: Babbling Baby

Babbling Baby is an experiment in tabula-rasa learning from observable
teacher behaviour. A small, randomly-initialised model — the
[pupil](#concept-pupil) — plays a turn-taking word-emission
[game](#concept-game) against a competent language model — the
[teacher](#concept-teacher) — where each emission earns a point only
if its bytes decode to a real word in the target language. The
dictionary is the [verifier](#concept-verifier).

The minimum claim under test is whether the prediction signal from
queryable teacher behaviour, paired with a grounded verifier, is
sufficient to teach a tabula-rasa model to produce word-shaped output
on commodity hardware.

[Babbling](#concept-babbling), not understanding. Sounds, not sentences.

---

## Who We're Building For

### Hero: Ed-as-researcher

I'm a solo founder who fell into a thought experiment that grew. A
long conversation with my AI coding agent turned into an architectural
sketch for a different way of building language models — bootstrapped
from observable teacher behaviour through play with a grounded
verifier, the way AlphaZero bootstraps game-play, rather than through
centralised pretraining on curated corpora.

The architectural ambition is large. The first concrete step shouldn't
be. I want to know whether a tabula-rasa model on my MacBook can learn
to emit real words by playing a turn-taking [game](#concept-game)
against a competent [teacher](#concept-teacher), with a dictionary
keeping score. Not whether it's better than pretraining. Not whether
it scales. Not whether the broader picture stands up. Just whether the
absolute first link of the chain holds.

If it does, I've earned the right to design v1. If it doesn't, I've
falsified the load-bearing assumption underneath the whole picture
cheaply, and learned something more useful than another round of
architecture talk.

So that:

- I know whether the prediction signal from a queryable teacher is
  rich enough to drive learning at all, before I spend any more time
  designing the layers on top of it.
- I have a concrete artefact running on my hardware that I can probe,
  break, and iterate on — rather than a paper architecture I keep
  refining in conversation.
- The first experiment is cheap enough to falsify in a day or two, so
  failure costs nothing and success earns the right to a v1.

Instead of:

- Designing layers of an architecture whose foundation hasn't been
  tested.
- Mistaking a corpus-shaped pretraining experiment for a test of the
  paradigm it pretends to be.
- Treating v0 as the final answer rather than the cheapest possible
  first probe.


---

## What People Need

### Goal: Show that a tabula-rasa pupil can learn to babble from a queryable teacher

I want to see a small, randomly-initialised [pupil](#concept-pupil),
after a tractable amount of interactive training against a competent
[teacher](#concept-teacher) and a dictionary
[verifier](#concept-verifier), reliably emit byte sequences that
decode to real words in English.

[Babbling](#concept-babbling), not understanding. Word-sounds, not
sentences. The bar is recognisable output structure, not capability,
not novelty, not transfer.

Fulfilled when:

- The pupil's dictionary-valid emission rate climbs meaningfully above
  its untrained baseline over the course of a training run.
- The pupil's emissions are distinguishable to a reader from the
  untrained baseline's gibberish — they look like words even when
  they aren't yet correct.
- Optionally, the pupil trained interactively reaches recognisable
  word emission faster or more robustly than the same architecture
  trained on a static corpus of equivalent teacher text.

I never:

- Need to manually curate a corpus ahead of training.
- Need to source human-labelled examples — the dictionary is the
  labeller.
- Need to evaluate by hand — every emission is scored by lookup.

Even when:

- The pupil's outputs are noisy or low-quality early on.
- The teacher's outputs occasionally drift outside the dictionary
  (rare words, proper nouns, neologisms).
- The dictionary doesn't contain every valid English word —
  unreliability at the margin is acceptable for v0.


---

## What Good Looks Like

### Quality: The experiment is honest about its scope

The spec, the code, and the writeup describe v0 as v0 — a probe of
the single riskiest assumption underneath a much larger architectural
picture. Not the architecture. Not a frontier-comparable system. The
first sound a baby makes.

When v0 succeeds, the spec says what's earned and what isn't. When
v0 fails, the spec says what was tested and what wasn't, so the
failure is informative.

Supports: "The first experiment is cheap enough to falsify in a day
or two, so failure costs nothing and success earns the right to a
v1."


### Quality: The verifier carries the meaning

The [verifier](#concept-verifier) is the only judge. No human
grading, no model-as-judge, no subjective evaluation. A byte sequence
decodes to a word in the dictionary, or it doesn't.

This keeps the experiment cheap, reproducible, and immune to Goodhart
on ambiguous criteria. If the paradigm can't be made to work with a
perfectly grounded verifier, no fuzzier verifier will save it.

Supports: "Need to evaluate by hand — every emission is scored by
lookup."


### Quality: The pupil runs on the researcher's machine

The whole experiment fits on a single M2 Max MacBook with 96GB
unified memory. No cloud, no distributed training, no specialised
hardware. One person, one machine, one teacher API key, one
dictionary.

If v0 needs more than this, v0 is wrong-sized. The broader paradigm
makes democratisation a load-bearing claim; v0 should embody it from
the first commit.


### Quality: Progress is visible while it happens

The researcher should be able to watch learning unfold during a
training run, not only see a final number afterwards. The progress
trace — what fraction of emitted bytes are printable, then letters,
then plausible word-shapes, then dictionary words — makes the
direction of learning visible even before the final bar is met.

If learning fails, the trace says where it stalled. If it succeeds,
the trace says how it got there.


---

## What Is Always True

### Property: The pupil starts from random initialisation

For any run of the experiment,

- the [pupil](#concept-pupil)'s weights are drawn from a standard
  initialisation distribution (Kaiming, Xavier, or framework default)
  at the start of the run.
- no pretrained weights are loaded.
- the pupil has no language priors except those implicit in its
  architecture — attention, positional encoding, byte-level
  embedding.


### Property: The substrate is byte-level

For any input to the [pupil](#concept-pupil) and any output from it,

- the unit is bytes (0–255), not BPE tokens or words.
- the pupil predicts the next byte, not the next token.
- whether a sequence of bytes forms a word is decided by the
  [verifier](#concept-verifier), not by the tokeniser.

This is forward-compatible with modalities beyond text, where the
substrate is also bytes — audio bytes, image bytes, code bytes.


### Property: The corpus emerges from play, not curation

For any training run,

- training data is generated turn-by-turn during the
  [game](#concept-game), not collected in advance.
- the [teacher](#concept-teacher)'s outputs are produced live in
  response to the game's state, not sampled from a static corpus.
- the [pupil](#concept-pupil)'s own emissions, scored by the
  [verifier](#concept-verifier), are part of the training signal.

This is the structural distinction from pretraining-on-Claude-outputs:
the data exists because the game was played, not because someone
generated and shelved it. The AlphaZero analogy holds — the corpus
is the residue of play, not its precondition.


### Property: Each game starts with a random dictionary word

For any game session,

- the initial context — the seed word for the very first turn — is
  drawn uniformly at random from the [verifier](#concept-verifier)'s
  dictionary.
- the [pupil](#concept-pupil)'s exposure to language begins with a
  real word, not noise, so the first imitation signal is well-formed
  from turn zero.
- the seed word is logged with the run, making the session
  reproducible.

This avoids the cold-start problem where the pupil's first context
would otherwise be either empty or randomly-sampled bytes — neither
of which is useful as conditioning.


### Property: The verifier is grounded and trivial

For any emitted byte sequence,

- the [verifier](#concept-verifier) returns 1 if the sequence
  decodes to a word in the target-language dictionary, 0 otherwise.
- the dictionary is a known word list (e.g.,
  `/usr/share/dict/words`, NLTK's wordlists, `pyspellchecker`).
- no learned verifier, no model-as-judge, no human-in-the-loop.

The verifier's dumbness is a feature. Any gradient flowing from it
is unambiguous.


### Property: The teacher is queryable, not transcribed

For any turn in the [game](#concept-game),

- the [teacher](#concept-teacher)'s output is produced live in
  response to the current game state.
- the teacher's behaviour may be shaped by instructions or context
  provided by the experimenter.
- the teacher's identity (which model, which configuration) is logged
  but not constrained by the spec.

This preserves the live-interaction property that distinguishes the
paradigm from pretraining. Pre-recording the teacher's outputs and
training on the recording would collapse this back to corpus-shaped
training.


### Property: Every emission is recorded for evaluation

For any turn in any [game](#concept-game),

- the tuple `(player, context, emission, score, turn_index)` is
  recorded to the run's log.
- the log is durable across the session — kept on disk, not just in
  memory.
- the recorded set is the basis for partitioning emissions into
  *seen by the pupil* and *unseen by the pupil* for held-out
  evaluation.

The [pupil](#concept-pupil)'s ability to emit words it never saw the
[teacher](#concept-teacher) produce is a stronger claim than
in-distribution mimicry; the recording is what makes that claim
testable later without re-running the experiment.


### Property: Progress is visible at multiple granularities

For any training run, the following are logged at regular intervals:

- the proportion of emitted bytes that are printable ASCII.
- the proportion that are letters [a-zA-Z].
- the proportion of letter-runs (between non-letter chars) of
  plausible word length (3–12 characters).
- the proportion of those runs that are dictionary words.

The progress trace is how the researcher sees *how* learning happens
or fails to happen, not only whether it does.


---

## The Domain

### Concept: Pupil

The pupil is a small (10M–50M parameter) byte-level decoder-only
transformer, randomly initialised, trained interactively during the
[game](#concept-game). It runs on the researcher's local machine.

The pupil is the artefact being tested. Its capability at the end of
v0 is the experiment's result.


### Concept: Teacher

The teacher is a competent language model that produces valid words
on demand. For v0, the teacher is Claude, accessed via API. The
teacher's role is to emit reference behaviour the [pupil](#concept-pupil)
observes and predicts.

The teacher is not fine-tuned, customised, or held to any standard
beyond "produces real words competently." It is a black box from
the pupil's perspective — only its outputs are observable.


### Concept: Game

The game is a turn-taking word-emission protocol. On each turn, a
player (the [teacher](#concept-teacher) or the [pupil](#concept-pupil))
emits one word given some minimal context (the partner's last word,
a seed, or no context). The [verifier](#concept-verifier) scores the
emission. Scores accumulate; the stated objective is to maximise
score.

The game's purpose is to generate (state, emission, verdict) tuples
that serve as training signal. It is not designed to be fun,
semantically rich, or open-ended at v0.


### Concept: Verifier

The verifier is a dictionary lookup. It returns 1 if the emitted
byte sequence decodes to a word in the dictionary, 0 otherwise. The
dictionary is a fixed wordlist for the target language; for v0 the
target language is English.

The verifier is intentionally minimal. Its rigidity is the source
of its usefulness.


### Concept: Score

The score is the count of dictionary-valid words emitted by a player
across a [game](#concept-game) session. The [pupil](#concept-pupil)'s
score relative to the [teacher](#concept-teacher)'s, and relative to
its own past performance, is the primary metric.

Score is not a proxy for understanding or capability beyond
[babbling](#concept-babbling). A high-scoring pupil at v0 has
demonstrated word-shape acquisition, nothing more.


### Concept: Babbling

Babbling is the production of structurally valid output without
semantic intent. A baby babbling makes phonemes that resemble words
without meaning words. A [pupil](#concept-pupil) babbling emits byte
sequences that form real words without using them to communicate.

The bar for v0 is babbling, not speech. The pupil isn't expected to
respond meaningfully to its context, only to produce outputs whose
shape matches the language it has observed.
