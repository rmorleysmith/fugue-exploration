# The tidying standard

Every script in `scripts/` descends from a file in the frozen June 2026 archive
(`rmorleysmith/hsln-archive`, `python.zip`). The archive is the record of what was done; this
repository is the record of what it *means*. A script is admitted here only when it meets all
of the following.

## 1. The header docstring has exactly five parts

```
"""<Title: one line, what the script shows>

WHAT IT COMPUTES  — the objects, in standard notation, with the definitions used.
WHAT THE FIGURE SHOWS — panel by panel, in one sentence each.
WHAT IS PROVED / KNOWN — the classical theorem(s) behind it, with a reference, and the Lean
    name in rmorleysmith/fugue-formal where one exists (e.g. `Fugue.EulerCompose.prod_eulerFactor`).
WHAT IS NOT CLAIMED — the over-reading the script could invite, stated and refused.
HOW THIS WAS FOUND — the reasoning path from the June exploration: the pattern that was noticed,
    the physical picture it suggested, and what turned out to be the precise statement.
"""
```

The last part is the point of the repository. It is written in the first person plural, plainly,
and it names the lofty guess as a guess ("we imagined a residue as …; the precise statement is …").

## 2. Vocabulary

- "DNA" → "coefficient vector" (first use: "the Fourier coefficients, which the June files
  called the DNA"). "Neuron" → "zero" (same convention). "Gravity" for the Siegel off-diagonal →
  "coupling κ". "Genus" for a theta series → "degree".
- No "machine-checked", "proved", "theorem" for anything that is not a cited theorem or a named
  Lean declaration. Numerical agreement is "verified to 1e−12 on n ≤ N".
- No factoring, hashing, cryptography, Bitcoin, PGP, "bignum", "secret", "attack". The only
  factoring that remains is `factorize.py`, which is explicitly about *why composing divisibility
  detectors is trial division*.
- No names of people or companies other than cited mathematicians.
- No "[REDACTED NAME]", no first-person singular, no "user asked".

## 3. Code

- Runs from `scripts/` with `python <name>.py` under the pinned environment
  (`requirements.txt`), writes to `../figures/<NN>_<name>.png` with the June figure number
  kept where one existed, prints its checks, and exits 0. Every printed check is an assertion.
- Imports only stdlib, numpy, scipy, matplotlib, mpmath and the shared modules in `scripts/`
  (`style`, `modular`, `geom`, `coloring`, `e8`, `siegel`, `zeros`, `dna`, `dna_algebra`).
  Nothing imports a removed script.
- Keep the mathematics and the figure design; remove dead code, duplicated helpers and
  parameters that only existed to chase a number.

## 4. The family note

Each family has `notes/<family>.md`: a page that reads the family's figures in order, states
the bridges between them, and ends with a table *script → figure → theorem → Lean name →
what is not claimed*. It cites the evidence ledger of the audit (`rmorleysmith/fugue`,
`Fugue/02-threads/evidence-ledger.md`) by row id for every grade.

## 5. What is left out, and why

| Removed | Reason |
|---|---|
| `factor_*`, `hsln_bignum_*`, `hsln_factor_dna`, `hsln_prime_dna`, `hsln_cofactor_*`, `hsln_dna_matrix_factor`, `hsln_e8_torsion_factor`, `hsln_reverse_factor`, `hsln_tau_compose`, `phi_via_category` | factoring probes; their negative conclusions are recorded once, in the audit (ledger G-rows and appendix A), and need no figures |
| `hsln_hash_*`, `hsln_pgp_dance`, `hsln_bitcoin_*` | hash and key demonstrations; same |
| `hsln_hodge`, `hsln_ktheory`, `hsln_langlands`, `hsln_yangmills`, `hsln_quantum_state`, `hsln_cup_product`, `hsln_trace` | illustrate Lean files that state conjectures; deferred until the corresponding chapter of the exposition exists |
