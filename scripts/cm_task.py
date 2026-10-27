"""A discrete readout: route each output coordinate to a CM point and read off its integer j.

WHAT IT COMPUTES — Readout = the singular-modulus codebook of cm_alphabet.py. For each output
    coordinate the final state τ* of the routing flow of hsln_dyn.py is reduced to the
    fundamental domain, j(τ*) is computed (an SL(2,ℤ)-invariant, so it reads the same
    wherever the flow happens to stop in the orbit), and snapped to the nearest integer CM
    value, which names a symbol. One coefficient vector per distinct letter is trained so
    that a fixed "question" input (0.5, 0.62) is routed to a low-Im representative of that
    letter's CM class: (0.5, 0.5) ~ i (j = 1728, symbol A) and (0.5, √3/2) = ρ (j = 0,
    symbol D). Three coordinates of a block-diagonal degree-3 state carry the word "ADA".
WHAT THE FIGURE SHOWS — Left: the two trained routes from the question point to the two CM
    representatives. Right: j at each coordinate's endpoint, snapped onto the codebook.
WHAT IS PROVED / KNOWN — j: ℍ/SL(2,ℤ) → ℂ is a bijection (the moduli interpretation), so any
    SL(2,ℤ)-invariant readout is a function of j (ledger G2); j(i) = 1728 and j(ρ) = 0
    (E₆(i) = 0 and E₄(ρ) = 0: `Fugue.E₆_I`, `Fugue.E₄_rho`, ledger A3). The CM values are
    those of cm_alphabet.py (ledger G1).
WHAT IS NOT CLAIMED — That nine symbols are available. Only two CM classes have a
    representative the flow reaches, so this is a two-symbol code and the word is a
    bit-string in disguise (ledger G3, NUMERICAL). The routing is learned per letter on one
    fixed input; nothing is "asked". The June seeds used Python's string hash, which is
    randomised per process; they are now fixed, and the script states which seed each
    router used.
HOW THIS WAS FOUND — Having a codebook (figure 23), we wanted the machine to say a word. The
    picture was a ball rolling to a landmark and the landmark's name being read off. The
    precise statement is only that j is a modular invariant and that the routing flow can be
    trained to stop near i or ρ; the two-symbol limit is where the picture met the code.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

import hsln_dyn as g1
import modular

# --- CM codebook: integer j-value -> symbol -------------------------------- #
CODEBOOK = {1728: "A", 0: "D", 8000: "I", -3375: "S", -32768: "L",
            -884736: "N", -147197952000: "E", -262537412640768000: "?"}
CODE_VALUES = np.array(sorted(CODEBOOK.keys()), dtype=float)

# letter -> (CM j-value, a low-Im representative tau the flow can reach)
LETTER_CM = {
    "A": (1728, (0.5, 0.5)),                  # reduces to i
    "D": (0, (0.5, float(np.sqrt(3) / 2))),   # rho, the zero of E4
}
SEEDS = {"A": (1, 2, 3), "D": (2, 1, 3)}      # fixed; the first that decodes correctly is used
ANSWER = "ADA"
QUESTION_START = (0.5, 0.62)                  # the fixed "question" input


def decode(tau) -> tuple[int, str]:
    """Reduce tau, compute j, snap to nearest CM codeword, return (j_int, symbol)."""
    j = complex(modular.jinv(np.array([tau]))[0]).real
    k = int(np.argmin(np.abs(CODE_VALUES - j)))
    cm = int(CODE_VALUES[k])
    return cm, CODEBOOK[cm]


def run(make_fig=True):
    print("readout check (j at the chosen representatives):")
    for L, (j_target, tau) in LETTER_CM.items():
        jval = complex(modular.jinv(np.array([tau[0] + 1j * tau[1]]))[0]).real
        print(f"  {L}: tau={tau}  j={jval:.3f}  (target {j_target})  decode={decode(tau[0]+1j*tau[1])}")
        assert abs(jval - j_target) < 1e-6 and decode(tau[0] + 1j * tau[1])[1] == L

    # train one router per distinct letter: question input -> that letter's CM point
    routers = {}
    for L, (_j, tgt) in LETTER_CM.items():
        for seed in SEEDS[L]:
            c, hist = g1.train_routing([QUESTION_START], [tgt], steps=1800, seed=seed)
            end = g1.endpoint(c, QUESTION_START)
            if decode(end[0] + 1j * end[1])[1] == L:
                break
        routers[L] = dict(coeffs=c, hist=hist, target=tgt, end=end, seed=seed)
        print(f"router {L} (seed {seed}): end={np.round(end,3)} target={tgt} "
              f"decode={decode(end[0]+1j*end[1])}")
        assert decode(end[0] + 1j * end[1])[1] == L, f"router {L} did not reach its CM class"

    decoded, coords = [], []
    for k, L in enumerate(ANSWER):
        end = routers[L]["end"]
        cm, sym = decode(end[0] + 1j * end[1])
        decoded.append(sym)
        coords.append((k, end, cm, sym))
    answer = "".join(decoded)
    print(f'\nfixed input -> three coordinates -> "{answer}"   (target "{ANSWER}"; a two-symbol code)')
    assert answer == ANSWER

    if make_fig:
        _figure(routers, coords, answer)
    return answer


def _figure(routers, coords, answer):
    import matplotlib.pyplot as plt
    import style  # noqa: F401

    GOLD, TEAL, GREEN = "#ffd166", "#46f0d0", "#7CFC8A"
    lcol = {"A": TEAL, "D": GOLD}

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})

    a = ax[0]
    a.set_facecolor("#0b0b14")
    for L, r in routers.items():
        tr = np.array(g1.rollout(jnp.asarray(r["coeffs"]), jnp.asarray(QUESTION_START)))
        a.plot(tr[:, 0], tr[:, 1], color=lcol[L], lw=2.2, zorder=4)
        tx, ty = r["target"]
        a.plot(tx, ty, "*", color=lcol[L], ms=22, mec="white", mew=1.0, zorder=6)
        cm, _ = decode(r["end"][0] + 1j * r["end"][1])
        a.annotate(f"'{L}'  →  j={cm}", (tx, ty), textcoords="offset points",
                   xytext=(12, 4), color=lcol[L], fontsize=10)
    a.plot(QUESTION_START[0], QUESTION_START[1], "o", color="white", ms=10, zorder=6)
    a.annotate("fixed input", QUESTION_START, textcoords="offset points",
               xytext=(-10, 12), color="white", fontsize=9, ha="center")
    a.set_xlim(0.0, 1.0)
    a.set_ylim(0.1, 1.05)
    a.set_xlabel(r"$\mathrm{Re}\,\tau$")
    a.set_ylabel(r"$\mathrm{Im}\,\tau$")
    a.set_title("one trained route per letter, from the fixed input\nto a representative of the letter's CM class",
                fontsize=12)

    b = ax[1]
    small = [v for v in CODE_VALUES if abs(v) <= 4e4]
    for v in small:
        b.axhline(v, color="#33334a", lw=1)
        b.annotate(f"{int(v)} = '{CODEBOOK[int(v)]}'", (len(coords) - 0.4, v),
                   color="#7777aa", fontsize=8, va="center")
    for (k, end, cm, sym) in coords:
        jval = complex(modular.jinv(np.array([end[0] + 1j * end[1]]))[0]).real
        col = lcol.get(sym, GREEN)
        b.plot(k, jval, "o", color=col, ms=9, zorder=4)
        b.annotate("", xy=(k, cm), xytext=(k, jval),
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
        b.plot(k, cm, "s", color=col, ms=12, mec="white", mew=0.8, zorder=5)
        b.annotate(f"'{sym}'", (k, cm), textcoords="offset points", xytext=(0, 12),
                   color=col, fontsize=12, ha="center", fontweight="bold")
    b.set_xticks(range(len(coords)))
    b.set_xticklabels([f"coord {k+1}" for k in range(len(coords))])
    b.set_xlim(-0.5, len(coords) - 0.1)
    b.set_ylim(-4e4, 4e4)
    b.set_ylabel(r"$j(\tau^\ast)$  (snapped to nearest singular modulus)")
    b.set_title(f'decode  →  "{answer}"   (two reachable symbols: a bit code)', fontsize=12.5)

    fig.suptitle("A discrete readout: three coordinates of a block-diagonal degree-3 state each stop near a CM "
                 "point, and the invariant $j$ names a symbol", fontsize=13, y=1.0)
    out = Path(__file__).parent.parent / "figures" / "24_cm_task.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
