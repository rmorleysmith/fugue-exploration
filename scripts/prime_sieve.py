"""The Sieve of Eratosthenes as a product of four sine combs (figure 31).

WHAT IT COMPUTES
    For each p in {2, 3, 5, 7} the comb  s_p(x) = sin(πx/p),  which vanishes on the integers
    exactly at the multiples of p.  Their product  F(x) = ∏_p sin(πx/p)  vanishes at every
    integer that has a prime factor in {2, 3, 5, 7}.  Since every composite n ≤ 100 has a
    prime factor ≤ √100 = 10, the integers 11 ≤ n ≤ 100 with F(n) ≠ 0 are exactly the primes;
    the four sieving primes are added back by hand.  The survivors are compared with trial
    division.

WHAT THE FIGURE SHOWS
    Top left:  the four combs, one per sieving prime, offset vertically.
    Top right: |F(x)| on [2, 50]; gold dots mark the primes, which are the only integers
               above 7 where the product is non-zero.
    Bottom:    the integers 2..100, gold where the product survives (or the integer is a
               sieving prime); 25 of 25 primes, no composite.

WHAT IS PROVED / KNOWN
    This is the Sieve of Eratosthenes (c. 240 BC): sieving by the primes ≤ √N leaves the
    primes in (√N, N].  The product of combs is a way of evaluating "is n divisible by some
    p ≤ 7" in one expression; it adds nothing to the sieve.  Lean:
    `Fugue.sieve_2357` in `Fugue/SieveEta.lean` states that for 11 ≤ n ≤ 100, n is prime iff
    none of 2, 3, 5, 7 divides it (ledger D9).

WHAT IS NOT CLAIMED
    Not a new way of finding primes (ledger D10 refutes that reading).  Multiplying the combs
    is equivalent to trial division by {2, 3, 5, 7}; it does not scale past N = 100 without
    adding a comb for every prime up to √N, which is exactly what the sieve does.  Nothing here
    touches the distribution of primes, zeta, or its zeros.

HOW THIS WAS FOUND
    We had been composing modular forms by multiplying them, and the rule we kept seeing was
    "the zero set of a product is the union of the zero sets".  We asked what the simplest
    arithmetic use of that rule would be, and imagined each prime as a detector that "rings
    to zero" on its own multiples; multiplying the detectors unions their zeros, so the
    composites are killed and the primes survive.  The picture is correct, and the precise
    statement is the oldest one in the subject: it is Eratosthenes' sieve, written as a
    product of sines.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SIEVE = [2, 3, 5, 7]
M = 100


def field(x):
    x = np.asarray(x, float)
    out = np.ones_like(x)
    for p in SIEVE:
        out = out * np.sin(np.pi * x / p)
    return out


def is_prime(n):
    if n < 2:
        return False
    return all(n % d for d in range(2, int(n ** 0.5) + 1))


def run(make_fig=True):
    n = np.arange(2, M + 1)
    found = set(SIEVE) | {int(k) for k in n if k > max(SIEVE) and abs(field(k)) > 1e-9}
    truth = {int(k) for k in n if is_prime(k)}

    print(f"survivors of F(x) = prod_p sin(pi x/p), p in {SIEVE}, plus the sieving primes:")
    print("  " + "  ".join(str(p) for p in sorted(found)))
    ok = found == truth
    print(f"{len(found)} survivors;  primes <= {M} by trial division: {len(truth)};  "
          f"equal: {ok}")
    assert ok, f"missed {sorted(truth - found)}  extra {sorted(found - truth)}"

    # The comb vanishes on the integers exactly at the multiples of p.
    for p in SIEVE:
        zero = np.abs(np.sin(np.pi * n / p)) < 1e-9
        assert np.array_equal(zero, n % p == 0), f"comb for p={p} is not the multiples of p"
    print(f"each comb sin(pi x/p) vanishes on 2..{M} exactly at the multiples of p: True")

    if make_fig:
        _figure(n, found, truth)
    return ok


def _figure(n, found, truth):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    GOLD = "#ffd166"
    COLP = ["#46f0d0", "#ffd166", "#ff7b6b", "#c0a3ff"]

    fig = plt.figure(figsize=(15.5, 7.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.32)
    top = gs[0].subgridspec(1, 2, width_ratios=[1, 1.1])
    a = fig.add_subplot(top[0]); b = fig.add_subplot(top[1])
    c = fig.add_subplot(gs[1])

    # --- A: the four combs --- #
    xs = np.linspace(2, 30, 2000)
    for i, p in enumerate(SIEVE):
        a.plot(xs, np.sin(np.pi * xs / p) + 2.4 * i, color=COLP[i], lw=1.3)
        a.axhline(2.4 * i, color="#333", lw=0.5)
        a.text(2.2, 2.4 * i + 0.95, f"sin(πx/{p}): zero at multiples of {p}", color=COLP[i],
               fontsize=9)
    a.set_yticks([]); a.set_xlabel("integer x")
    a.set_title("four combs, one per sieving prime", fontsize=12)

    # --- B: the product; primes survive --- #
    xs2 = np.linspace(2, 50, 6000)
    b.plot(xs2, np.abs(field(xs2)), color="#5a5a72", lw=0.9, alpha=0.8)
    for k in range(2, 51):
        prime = k in truth
        b.plot(k, abs(field(k)), "o", color=GOLD if prime else "#444",
               ms=8 if prime else 5, mec="white" if prime else "none",
               mew=0.5, zorder=4 if prime else 3)
    b.set_xlabel("integer"); b.set_ylabel(r"$|F(x)|$ = $\prod_p |\sin(\pi x/p)|$")
    b.set_title("the product vanishes wherever some p | x;\n"
                "above 7 the survivors (gold) are the primes", fontsize=12)

    # --- C: the result, 2..M --- #
    for k in range(2, M + 1):
        prime = k in found
        c.plot(k, 0, "s", color=GOLD if prime else "#2a2a38",
               ms=11 if prime else 8, mec="white" if prime else "#444", mew=0.4)
        if prime:
            c.annotate(str(k), (k, 0), textcoords="offset points", xytext=(0, 9),
                       color=GOLD, fontsize=7.5, ha="center", rotation=90)
    c.set_ylim(-0.5, 1.2); c.set_yticks([]); c.set_xlabel("integer")
    c.set_title(f"survivors ≤ {M} plus the sieving primes: {len(found)}, "
                f"equal to the {len(truth)} primes by trial division", fontsize=12)

    fig.suptitle("The Sieve of Eratosthenes as a product of sine combs: sieving by the primes "
                 "≤ √100 leaves exactly the primes ≤ 100", fontsize=13, y=0.99)
    out = Path(__file__).parent.parent / "figures" / "31_prime_sieve.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
