# Prompt Book

A log of the significant prompts that built this project, the context they were
issued in, what came back, and what we learned. Not every keystroke is here —
just the prompts that shaped a decision, unblocked a step, or set a pattern
that later prompts inherited.

The project was developed under the *Orchestration of AI Agents* course, so
the workflow itself was the assignment: a small set of human-authored framing
prompts, and a much larger set of machine-generated per-issue prompts driven by
those frames.

---

## 1. How this repo was built

We did not pair-program with the AI line by line. Instead the workflow was:

```
human → PRD / PLAN / TODO       (high-level intent)
TODO → GitHub issues            (one issue per row in the TODO table)
issue → "issue worker" agent    (Claude, one issue per branch + PR)
PR    → CI + code review        (ruff, pytest --cov, human review, merge)
```

So most prompts in this book fall into three buckets:

1. **Framing prompts** — written by a human, *once*, to produce the PRD,
   architecture plan, and the phased TODO. These set the language and
   constraints the rest of the project speaks in.
2. **Dispatch prompts** — short, structured, machine-templated prompts handed
   to the issue worker agent for each TODO row. Boring on purpose.
3. **Repair / refactor prompts** — written by a human when CI failed, when a
   review surfaced a problem, or when an architectural assumption needed to
   change mid-flight.

The dispatch prompts are repetitive (by design) so they are summarized as a
single template in §3 rather than listed one-by-one. The interesting prompts
are in §2 and §4.

---

## 2. Framing prompts (one-shot, high-leverage)

### 2.1 — "Write the PRD for HW1"

**Context:** very first prompt. The course handed out a Hebrew/English PDF
specifying the denoising task at a high level (4 sine waves, 10 ms windows,
compare MLP / RNN / LSTM). Nothing was committed yet except the PDF and a
`hello world` `main.py`.

**Goal:** turn the spec into a self-contained PRD that the rest of the work
could be derived from — measurable goals, non-goals, functional requirements,
success criteria, open questions.

**Prompt (paraphrased):**

> Read `docs/file.pdf` (the course assignment). Produce `docs/PRD.md` for the
> sine-wave denoising project. Use the structure: Overview, Problem Statement,
> Goals (measurable + quality bars), Non-Goals, Functional Requirements,
> Success Criteria, Open Questions. Reflect the course constraints exactly —
> 4 sine waves, 10 s @ 1000 Hz, per-sample Gaussian noise, one-hot selector +
> 10-sample window in, 10-sample clean window out. Be opinionated about the
> non-goals.

**What came back:** `docs/PRD.md` close to the version that shipped. Two
things needed iteration:

- The first draft hand-waved the input shape ("the model sees the noisy
  context and the selector"). We pushed back asking for an exact length,
  which is how F5 (`length 14`) and ADR-003 (sequence layout for RNN/LSTM)
  came to exist.
- The first draft listed Transformers as a "stretch goal". We asked for a
  ruthless non-goals list; Transformers moved to *backlog*.

**Lesson:** when you ask for a PRD, force the model to commit to numbers.
"Vector of length 14" is testable; "selector and window" is not.

### 2.2 — "Write the architecture plan and the model spec"

**Context:** PRD existed. We needed a `PLAN.md` (architecture, ADRs, data
schemas) and a `PRD_models.md` (per-model layer-by-layer spec) before any
code was written.

**Goal:** lock the public surface (`SDK` class), the package layout, and the
config schema *before* implementing anything, so that 30 follow-up issues
could be scoped without re-litigating shape decisions.

**Prompt (paraphrased):**

> Given `docs/PRD.md`, write `docs/PLAN.md`. Include a C4-style context /
> container / component breakdown, a data-flow diagram for one training run,
> the JSON schema of `config/default.json` with concrete defaults, and at
> least 5 ADRs covering: framework choice, how the selector is fused into the
> input, how RNN/LSTM consume the window, what the public API surface is, and
> the config format. Then write `docs/PRD_models.md` with one section per
> model (MLP / RNN / LSTM): layer-by-layer description, parameter count for
> the default config, and the exact config keys each model accepts.

**What came back:** `docs/PLAN.md` and `docs/PRD_models.md` close to what
shipped. ADR-002 (concatenation vs. FiLM) and ADR-003 (sequence layout) came
straight out of forcing the model to write down *why*, not just *what*.

**Lesson:** ADRs are cheap to ask for and disproportionately useful later.
When a downstream issue asks "why is the selector broadcast per timestep?",
the answer is one grep away.

### 2.3 — "Turn the PRD and PLAN into a phased TODO"

**Context:** PRD + PLAN existed. We needed an executable backlog.

**Goal:** produce one TODO row per atomic, ship-in-one-PR unit of work, with
a Definition of Done that a CI pipeline could verify.

**Prompt (paraphrased):**

> Read `docs/PRD.md` and `docs/PLAN.md`. Produce `docs/TODO.md` as a phased
> task list: Phase 0 scaffolding, Phase 1 data, Phase 2 models, Phase 3
> training & evaluation, Phase 4 analysis & plots, Phase 5 polish &
> submission. Each row must have an owner column, a one-line task, and a
> concrete Definition of Done that can be checked by reading the diff or
> running a command. Cap each row at one PR's worth of work. No row should
> say "polish" without being specific about *what* counts as polished.

**What came back:** the file at `docs/TODO.md`. The DoDs were the most
valuable output — they became the Definition-of-Done block on every later
GitHub issue, and they were what made the dispatch prompt in §3 work at all.

**Lesson:** "list the tasks" gives you a wishlist; "list the tasks and how
you would prove each one is done" gives you a backlog.

### 2.4 — "Open one GitHub issue per TODO row"

**Context:** TODO existed. We wanted GitHub issues so the issue worker agent
could pick them up.

**Goal:** mechanical translation from TODO rows to issues, with consistent
labels and bodies, so the dispatch prompt in §3 could be templated.

**Prompt (paraphrased):**

> For every row in `docs/TODO.md` whose status is ☐, open a GitHub issue
> using `gh issue create`. Title format: `<task#> <task>`. Body format must
> include the phase, the task #, the task text, the DoD, and a link back to
> `docs/TODO.md`. Apply a label per phase (`phase-0` … `phase-5`). Do not
> open issues for rows already marked ☑.

**What came back:** issues #3 through #33, each with the same body shape
(phase / task # / task / DoD / link). That uniformity is what lets the
dispatch prompt be one template instead of thirty.

**Lesson:** if you intend to drive dozens of follow-up runs from a single
template, spend the prompt budget upfront on making the *inputs* to that
template uniform.

---

## 3. The dispatch prompt (templated, ~30 invocations)

Every implementation PR in Phases 0–4 was opened by an issue worker agent
running a near-identical prompt. The template is short on purpose:

```
You are an issue worker agent. Your job:
1. Read issue #<N> using `gh issue view <N>`.
2. Implement the work described in the issue. Follow existing code patterns
   and conventions.
3. Commit your changes with a clear message referencing the issue.
4. Push your branch (`issue-<N>-<timestamp>`) and open a pull request with
   `gh pr create`. The PR body must include "Closes #<N>" so the issue
   auto-closes when merged.

Rules: implement only ONE issue. Do NOT modify unrelated code. Keep the diff
focused and reviewable. The build, linting, and tests must all pass before
you commit.
```

**Why so short.** All the *real* requirements live in the issue body
(phase, task, DoD) and in the repo it lands in (PRD, PLAN, ruff config,
existing modules to mirror). The dispatch prompt's job is just to point the
agent at the right issue and the right rules of engagement.

**Examples of how the same template produced very different work.**

| Issue | What the agent inferred from the repo alone |
|---|---|
| #7 `data/signals.py` | Wrote pure NumPy generator, derived shape `(4, 10000)` from PRD F1–F2, mirrored existing `tests/unit/data/` layout for tests. |
| #15 `models/lstm.py` | Followed the layer spec in `PRD_models.md` §4, produced `(B, 10)` output, matched the kwarg names (`hidden_size`, `num_layers`, `dropout`, `bidirectional`) from `config/default.json`'s `model.lstm` block. |
| #22 `train.py` CLI | Reused the existing `SDK` class, did not re-implement the training loop. Argparse flags matched the README table. |
| #26 `notebooks/results_analysis.ipynb` | Used only the public `SDK` surface (per ADR-004), did not import internals. |

**Iterative improvements to the template (in chronological order):**

1. **Initial version** had only steps 1–3. Several PRs landed without
   "Closes #N", so issues stayed open after merge. Added the explicit
   instruction to include `Closes #<N>` in step 4.
2. **Second version** missed a CI check on one PR (the agent committed before
   running tests). Added the explicit "build, linting, and tests must all
   pass before you commit" rule, plus an explicit "no half-done
   implementations, no TODOs, no placeholder code".
3. **Third version** had two PRs touch unrelated files in a single diff
   (drive-by ruff fixes during a model PR). Added "do NOT modify unrelated
   code". Drive-by cleanups now go to their own dedicated issue (#29 was the
   ruff sweep, #30 was the coverage gate).

**Lesson.** The dispatch prompt is a contract between the human reviewer and
the agent. Every time a PR slipped through review with a problem, the cure
was to add one sentence to the template, not to write a longer per-issue
prompt.

---

## 4. Repair / refactor prompts (one-off, surgical)

These are the prompts that fixed something the dispatch loop got wrong.

### 4.1 — "Make ruff clean across the repo" (issue #29)

**Context:** by the end of Phase 4 we had ~20 source files, written by ~20
independent agent runs. `ruff check .` produced a long list of unused
imports, mixed quote styles, missing trailing commas, and a few `B008`
mutable-default-argument warnings.

**Prompt (paraphrased):**

> Configure `ruff` in `pyproject.toml` with a sensible rule set for this
> project (E, F, I, B, UP, SIM at minimum). Then run `uv run ruff check .`
> and fix every violation across the repo. Do not change behavior — only
> formatting, imports, and trivial idiom upgrades. Re-run the test suite to
> confirm nothing broke.

**What came back:** the commit on `main` titled *"Configure ruff and clean
up lints across the repo (issue #29)"*. The diff was big but mechanical.

**Lesson.** Lint cleanups should be one PR, late, after the modules have
settled. Trying to enforce strict lint from issue #1 would have produced
endless trivial back-and-forth on every model PR.

### 4.2 — "Enforce 85 % coverage in CI" (issue #30)

**Context:** PRD §3 quality bar was ≥85 % coverage. We had been *reporting*
coverage in CI but not *failing* the build on it. A few modules had drifted
to ~78 %.

**Prompt (paraphrased):**

> Add `--cov-fail-under=85` to the pytest invocation in `.github/workflows/ci.yml`.
> Then identify modules below 85 % coverage and add the smallest possible
> tests to bring the whole repo above the gate. Prefer testing public
> behavior over internals. Do not lower the gate.

**What came back:** the merged PR for #30. Two interesting follow-ups
fell out of the human review:

- The agent's first attempt added tests that *imported* uncovered code
  paths without asserting on them, which inflated coverage without
  exercising behavior. Review pushed back; the second attempt asserted on
  outputs.
- One module was genuinely hard to cover (a small CLI shim). We accepted
  it and let other modules carry the average rather than write a brittle
  subprocess test.

**Lesson.** "Add tests until coverage hits N" is a dangerous prompt — it
incentivizes the agent to inflate the metric. Always pair coverage prompts
with "prefer testing public behavior over internals", and review the *new
asserts*, not the new coverage number.

### 4.3 — "150-line file rule" (issue #31, in flight)

**Context:** submission guidelines cap each source file at 150
non-blank/non-comment lines. A few modules have crept over.

**Prompt (paraphrased):**

> Write `scripts/check_line_count.sh` that fails if any file under `src/`
> exceeds 150 lines (excluding blank and comment-only lines). Wire it into
> CI. Then split any over-budget file along natural seams — do not
> introduce new abstractions just to hide line count. Each split must
> preserve public imports.

**Lesson (anticipated).** The temptation under a hard line cap is to extract
helpers that do nothing but appease the linter. The prompt explicitly
forbids that, because we would rather have a slightly-too-long file than a
spurious `_helpers.py` that exists only to satisfy a script.

### 4.4 — Mid-flight refactor: "Make the SDK the only public surface"

**Context:** issues #17–#20 (training loop, optimizer factory, metrics,
robustness) were implemented before the SDK existed. The notebook in #26 was
about to import `sine_denoiser.training.loop.fit` directly. ADR-004 says the
SDK is the only public surface.

**Prompt (paraphrased):**

> Before issue #26 lands, finish issue #21 (`sdk.py`). The notebook must
> import only `from sine_denoiser import SDK`. If the SDK is missing any
> method the notebook needs (`generate_data`, `train`, `evaluate`,
> `predict`), add it now. Do not let the notebook reach into internals — if
> it would have to, that is a bug in the SDK, not in the notebook.

**What came back:** issue #21 landed first, the SDK grew the methods the
notebook needed, and #26 went in clean.

**Lesson.** Architectural rules (ADRs) only hold if you actively schedule
the work that enforces them *before* the work that would violate them.
The dispatch loop will happily honor the issue ordering you give it; if you
don't order issues to respect the architecture, the agent won't notice.

### 4.5 — "Polish the README" (issue #28)

**Context:** the README from Phase 0 was a stub. By the end of Phase 4 the
project had a CLI, an SDK, a config schema, plotting helpers, and a
robustness sweep — none of which were documented for a cold reader.

**Prompt (paraphrased):**

> Rewrite `README.md` so a cold reader can install, run, and extend the
> project after reading only the README. Sections: Task, Install (using
> `uv` only — no pip), Quickstart (one CLI invocation), CLI options table,
> SDK example (Python snippet), Configuration (one table per config block:
> `data`, `model`, `training`), Plots (snippets for each helper), Project
> layout, Development, License. Match the actual code — read
> `config/default.json`, `train.py`, and `sdk.py` and document what's
> there, not what you think should be there.

**What came back:** the README that ships on `main`. The "match the actual
code" clause caught two drift bugs: a CLI flag in an early draft did not
exist, and the SDK example used a kwarg name that had been renamed.

**Lesson.** Documentation prompts must explicitly point the model at the
code it should be documenting. "Document the SDK" produces a plausible
fiction; "read `sdk.py` and document its public methods" produces a
reference.

---

## 5. Best practices distilled

After ~30 issue-driven PRs and a handful of repair prompts, a few patterns
keep paying for themselves:

1. **Spend prompt budget upfront, on PRD / PLAN / TODO.** The framing
   prompts in §2 are the leverage. Every dispatch prompt in §3 is short
   only because the framing was thorough. If you find yourself writing long
   per-task prompts, the framing was probably under-specified.

2. **Definitions of Done are the unit of trust.** Every TODO row carries a
   one-sentence DoD that a reviewer (or CI) can check. The agent is allowed
   to interpret the *task*; it is not allowed to redefine *done*.

3. **Make the dispatch prompt boring.** Same template, same rules of
   engagement, same output shape. If a single PR needs special handling,
   that is a signal to either (a) split the issue or (b) handle it as a
   one-off repair prompt, *not* to fork the dispatch template.

4. **Patch the template, not the run.** Every time a PR slipped through with
   a problem, we added one sentence to the dispatch template instead of
   writing a longer prompt for the next issue. The template grew by 5
   sentences over ~30 runs; the per-issue prompts stayed the same length.

5. **Lint, coverage, and line-count gates land late, in dedicated PRs.**
   Phase 5 issues #29–#31 exist because we deferred those gates until the
   modules had settled. Trying to enforce them from issue #1 would have
   meant 30 trivial bikeshed PRs.

6. **Architectural rules only hold if the schedule respects them.** ADR-004
   ("SDK is the only public surface") only survived because we scheduled
   the SDK PR (#21) *before* the notebook PR (#26). If we had scheduled
   them the other way around, the notebook would have reached into
   internals and the ADR would have quietly died.

7. **For docs, point the model at the code.** "Document X" produces
   plausible fiction. "Read `<file>` and document the public methods you
   find there" produces a reference that matches reality.

8. **Coverage prompts are dangerous on their own.** Always pair them with
   "prefer testing public behavior over internals" and review the new
   asserts, not the new coverage number. Otherwise the agent will hit the
   gate by importing uncovered code without exercising it.

9. **Keep one prompt → one PR.** Every dispatch run produces exactly one
   branch and one PR. This makes review tractable, makes blame attributable,
   and means a bad PR can be closed without unwinding work on other issues.

10. **Write down the lessons.** This document exists because the patterns
    above are not obvious from the merged code. Future contributors (human
    or agent) inherit the *artifacts* easily; they inherit the *workflow*
    only if it is written down.
