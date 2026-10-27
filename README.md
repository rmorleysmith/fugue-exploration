# Fugue — the exploration, tidied

The June 2026 exploration (`rmorleysmith/hsln-archive`, frozen) produced 79 scripts and 65
figures while trying to build a neural network whose parameters are the Fourier coefficients
of a modular form. This repository is the same exploration with the frenzy removed: every
script that survives carries a header saying what it computes, what the figure shows, which
classical theorem and which Lean declaration stand behind it, what it does **not** claim, and
how the idea was found. The standard is [`TIDYING.md`](TIDYING.md); what was left out, and
why, is its last section.

| Family | Scripts | Figures | Note |
|---|---|---|---|
| Foundations | `render.py` and the shared modules | 01–13 | [`notes/foundations.md`](notes/foundations.md) |
| Spectral and Fourier | `spectral`, `explicit`, `hsln_spectral`, `lie_derivative`, `hsln_lie_landscape`, … | 34, 35, 38, 41, 42, 56, 63 | [`notes/spectral.md`](notes/spectral.md) |
| Composition and number theory | `prime_sieve`, `zeta_compose`, `factorize`, `landscape`, `lfunctions`, `bsd`, `eta_compose`, `hsln_two_rings`, `hsln_hecke_exercise` | 21, 31–33, 36, 37, 39, 40, 43 | [`notes/composition.md`](notes/composition.md) |
| The network | `poc*`, `hsln`, `hsln_dyn`, `hsln_genus2`, `coupling_sweep`, `cm_*`, `timeseries`, `scaling`, `skill_compose`, `grok` | 14–30, 44, 45 | [`notes/network.md`](notes/network.md) |

Run any script from `scripts/` with the pinned environment (`pip install -r requirements.txt`);
it prints its checks, writes its figure to `figures/`, and exits 0.

## How AI was used

The ideas in this work are Richard Morley-Smith's: the questions, the links between areas of
mathematics, and the choices of what to pursue. AI was used as a tool, in two phases.

- **May–June 2026, the original exploration.** The exploration was carried out in conversation
  with AI coding assistants, chiefly Anthropic's Claude through Claude Code (the design document
  also quotes Google's Gemini). The assistants wrote most of the code, Lean files and documents
  from the author's ideas and direction, quickly and without later editing. Those files are kept
  unchanged in [hsln-archive](https://github.com/rmorleysmith/hsln-archive).
- **October 2026 onwards.** Claude, through Claude Code, helps the author formalise the ideas in
  Lean 4, audit and tidy the earlier files, and write out the working: the steps between an
  intuition and a precise statement.

Checking the results does not rely on trusting the AI. Every Lean theorem is verified by the
Lean kernel against Mathlib, and `#print axioms` shows only Lean's three standard axioms. Every
script asserts the numbers it reports and can be rerun. Every grade in the audit cites the
evidence it rests on.

Companion repositories: the frozen originals and their timestamps
([hsln-archive](https://github.com/rmorleysmith/hsln-archive)), the evidence-graded audit
([fugue](https://github.com/rmorleysmith/fugue)), the Lean formalisation
([fugue-formal](https://github.com/rmorleysmith/fugue-formal)).

Licence: code Apache-2.0, text and figures CC-BY-4.0.
