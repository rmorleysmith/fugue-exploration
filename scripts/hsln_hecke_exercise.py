"""The prime coefficients of Δ determine all the others: Hecke's recursion exercised on τ(n).

WHAT IT COMPUTES
    Ramanujan's τ(n), the coefficients of Δ = q ∏_{n≥1} (1 − qⁿ)²⁴ ∈ S₁₂(SL₂(ℤ)), for
    n < 2600, as exact integers from the product.  Then the two rules that make Δ a Hecke
    eigenform,
        (1)  τ(p^{r+2}) = τ(p) τ(p^{r+1}) − p¹¹ τ(p^r),
        (2)  τ(mn) = τ(m) τ(n)  for gcd(m, n) = 1,
    are checked on the computed values, and τ(N) for composite N is rebuilt from the values
    τ(p) at the primes of N alone.  As a further check, τ(n) ≡ σ₁₁(n) (mod 691) for n < 2600.
    No figure.

WHAT THE FIGURE SHOWS
    None; the script prints its checks.

WHAT IS PROVED / KNOWN
    Mordell (1917): τ is multiplicative and satisfies the recursion; Hecke (1937): these are
    the eigenform identities T_p Δ = τ(p) Δ, and a normalised eigenform's coefficient vector is
    determined by its values at the primes (Serre, "A Course in Arithmetic", ch. VII, §5).
    Lean, `Fugue/Tau691.lean` and `Fugue/RamanujanL.lean`: `Fugue.Tau691.τ_int` (τ(n) ∈ ℤ, from
    `qExpansion_discriminant_eq`), `tau_congr_691` (τ(n) ≡ σ₁₁(n) mod 691),
    `Fugue.RamanujanL.τ_two`, `τ_three` (τ(2) = −24, τ(3) = 252), and the entire completed
    L-function with `ramanujanΛ_functional_equation`.  Mordell's multiplicativity and the
    recursion are not formalised there.  Ledger H14.

WHAT IS NOT CLAIMED
    The theorem is a determination: it reads the primes of N and produces τ(N).  It is not an
    inversion.  Run backward it does not produce the primes of N from τ(N): it only turns one
    question about N into a question about the much larger integer τ(N), and the map
    τ(p) ↦ p is not even injective in weight 2 (ledger H14); the June script's attempt at
    that direction is dropped.  Lehmer's conjecture (τ(n) ≠ 0) is open and untouched.

HOW THIS WAS FOUND
    We had been reading a coefficient vector as a complete description of a form, and asked
    how much of it is redundant.  The answer for a Hecke eigenform is "everything beyond the
    primes", and we wanted to see that with real integers rather than as a statement, so we
    rebuilt τ(360), τ(1000), τ(2491) from τ(2), τ(3), τ(5), τ(47), τ(53) and watched them come
    out exactly.  The lofty guess was that the rule might run the other way; the precise
    statement is Hecke's, which runs one way only.
"""
from __future__ import annotations

from math import comb, isqrt

import numpy as np

K = 12                      # weight of Δ; the recursion uses p^{k−1} = p^11
M = 2600                    # τ(n) is computed for 1 ≤ n < M


def build_tau(M: int) -> dict[int, int]:
    """τ(n) for 1 ≤ n < M from Δ = q·∏(1 − qⁿ)²⁴, as exact integers."""
    c = np.zeros(M, dtype=object)      # coefficients of ∏(1 − qⁿ)²⁴
    c[0] = 1
    for n in range(1, M):
        newc = np.zeros(M, dtype=object)
        for j in range(25):
            shift = j * n
            if shift >= M:
                break
            newc[shift:] = newc[shift:] + comb(24, j) * (-1) ** j * c[:M - shift]
        c = newc
    return {n: int(c[n - 1]) for n in range(1, M)}        # τ(n) = coefficient of q^n


def is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, isqrt(n) + 1))


def prime_powers(n: int) -> dict[int, int]:
    """The prime-power decomposition {p: e} of n, by trial division (the theorem is indexed by it)."""
    f, d = {}, 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


def sigma11(n: int) -> int:
    return sum(d ** 11 for d in range(1, n + 1) if n % d == 0)


def main():
    print(f"tau(n) for n < {M} from the product q prod (1 - q^n)^24 ...")
    tau = build_tau(M)

    known = {1: 1, 2: -24, 3: 252, 4: -1472, 5: 4830, 6: -6048, 7: -16744,
             8: 84480, 11: 534612, 12: -370944}
    ok = all(tau[n] == v for n, v in known.items())
    print(f"  tau agrees with the tabulated values (tau(2) = -24, tau(11) = 534612, ...): {ok}")
    assert ok

    cong = all((tau[n] - sigma11(n)) % 691 == 0 for n in range(1, M))
    print(f"  tau(n) == sigma_11(n) (mod 691) for n < {M}: {cong}   (Fugue.Tau691.tau_congr_691)")
    assert cong

    print("\n[A] Hecke recursion at primes:  tau(p^2) = tau(p)^2 - p^11")
    for p in [2, 3, 5, 7, 11, 13]:
        lhs, rhs = tau[p * p], tau[p] ** 2 - p ** 11
        print(f"    p={p:2d}:  tau(p^2) = {lhs:>18d}   tau(p)^2 - p^11 = {rhs:>18d}   {lhs == rhs}")
        assert lhs == rhs

    print("\n[B] multiplicativity:  tau(pq) = tau(p) tau(q) for distinct primes")
    for p, q in [(2, 3), (3, 5), (5, 7), (11, 13), (47, 53)]:
        lhs, rhs = tau[p * q], tau[p] * tau[q]
        print(f"    ({p:2d},{q:2d}):  tau(pq) = {lhs:>22d}   tau(p) tau(q) = {rhs:>22d}   {lhs == rhs}")
        assert lhs == rhs

    def from_primes(n: int) -> int:
        """tau(n) from the values tau(p) at the primes of n, by (1) and (2) only."""
        result = 1
        for p, e in prime_powers(n).items():
            tp = [1, tau[p]]
            for r in range(2, e + 1):
                tp.append(tau[p] * tp[r - 1] - p ** (K - 1) * tp[r - 2])
            result *= tp[e]
        return result

    print("\n[C] tau(N) rebuilt from the prime values alone (the content of the theorem):")
    for n in [12, 360, 1000, 2491]:
        rec = from_primes(n)
        print(f"    N={n:<5d} primes {prime_powers(n)}")
        print(f"          from primes: {rec}")
        print(f"          direct:      {tau[n]}      equal: {rec == tau[n]}")
        assert rec == tau[n]

    # every composite n < M, not just the four shown
    all_ok = all(from_primes(n) == tau[n] for n in range(2, M) if not is_prime(n))
    print(f"\n    rebuilt tau(n) equals the direct value for every composite n < {M}: {all_ok}")
    assert all_ok

    print("\nWhat this is: a determination, primes -> all coefficients, one way only.")


if __name__ == "__main__":
    main()
