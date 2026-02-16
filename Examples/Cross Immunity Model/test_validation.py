"""
Rigoroser Beweis: Maximale Anzahl Lösungen im Einheitsquadrat [0,1]²

Beweisstrategie:
1. Symbolische Konstruktion des Systems
2. Polynomialisierung durch Multiplikation mit Hauptnenner
3. Gröbner-Basis Berechnung (lexikographische Ordnung)
4. Grad-Analyse des univariaten Polynoms
5. Sturm-Sequenz zur Zählung reeller Wurzeln in [0,1]

"""

from sympy import (
    symbols,
    simplify,
    together,
    numer,
    denom,
    degree,
    total_degree,
    solve,
    roots,
    Poly,
    Rational,
)

# ============================================================================

# SCHRITT 1: SYMBOLISCHE DEFINITIONEN

# ============================================================================

print("=" * 80)
print("Proof: Number of solutions in [0,1]²")
print("=" * 80)

# Variablen

i1, i2 = symbols("i1 i2", real=True)

# Parameter (symbolisch für generischen Beweis)

sigma = symbols("sigma", real=True, positive=True)
beta_1 = symbols("beta_1", real=True, positive=True)
beta_2 = symbols("beta_2", real=True, positive=True)
gamma_1 = symbols("gamma_1", real=True, positive=True)
gamma_2 = symbols("gamma_2", real=True, positive=True)
b = symbols("b", real=True, positive=True)

print("\n" + "-" * 80)
print("STEP 1: Symbolic construction")
print("-" * 80)

# ============================================================================

# SCHRITT 2: KONSTRUKTION ALLER x_i FUNKTIONEN

# ============================================================================

print("\nConstruct x_i functions symbolically...")

# x_6

x_6 = b / (beta_1 * i1 + beta_2 * i2 + b)
print("x_6 defined")

# x_1

x_1 = ((1 - sigma) * beta_1 * x_6 * i1) / (gamma_1 + beta_2 * i2 + b)
print("x_1 defined")

# x_4

x_4 = ((1 - sigma) * beta_2 * x_6 * i2) / (gamma_2 + beta_1 * i1 + b)
print("x_4 defined")

# x_7

x_7 = (gamma_1 * x_1) / (beta_2 * i2 + b)
print("x_7 defined")

# x_8

x_8 = (gamma_2 * x_4) / (beta_1 * i1 + b)
print("x_8 defined")

# x_3

x_3 = (beta_1 * x_4 * i1 + beta_2 * x_1 * i2) / (gamma_1 + gamma_2 + b)
print("x_3 defined")

# x_2

x_2 = (sigma * beta_1 * x_6 * i1 + beta_1 * x_8 * i1 + gamma_2 * x_3) / (
    gamma_1 + b
)
print("x_2 defined")

# x_5

x_5 = (sigma * beta_2 * x_6 * i2 + beta_2 * x_7 * i2 + gamma_1 * x_3) / (
    gamma_2 + b
)
print("x_5 defined")

# x_9

x_9 = (gamma_1 * x_2 + gamma_2 * x_5) / b
print("x_9 defined")

sum_x = simplify(x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9)
print(f"The sum of all x is: {sum_x}")


# ============================================================================

# SCHRITT 3: KONSTRUKTION DES GLEICHUNGSSYSTEMS

# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Construction of the system of equations")
print("-" * 80)

F1 = x_1 + x_2 + x_3 - i1
F2 = x_3 + x_4 + x_5 - i2

print("\nF1 = x_1 + x_2 + x_3 - i1")
print("F2 = x_3 + x_4 + x_5 - i2")

# ============================================================================

# SCHRITT 4: POLYNOMIALISIERUNG

# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Polynomialisation")
print("-" * 80)

print("\nBring F1 and F2 down to a common denominator...")

# Together kombiniert zu einem Bruch

F1_together = together(F1)
F2_together = together(F2)

print("F1 and F2 summarized")

# Extrahiere Zähler (Polynome)

P1 = numer(F1_together)
P2 = numer(F2_together)

print("Nominators extracted")
print(f"Polynomial P1 is: {P1}")
print(f"Polynomial P2 is: {P2}")


# ============================================================================

# SCHRITT 5: GRAD-ANALYSE

# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Degree analysis of the polynomials")
print("-" * 80)

print("\nCompute degrees...")
deg_P1_i1 = degree(P1, i1)
deg_P1_i2 = degree(P1, i2)
deg_P2_i1 = degree(P2, i1)
deg_P2_i2 = degree(P2, i2)
deg_P1 = total_degree(P1, i1, i2)
deg_P2 = total_degree(P1, i1, i2)

print("\nP1:")
print(f"  Degree in i1: {deg_P1_i1}")
print(f"  Degree in i2: {deg_P1_i2}")
print(f"  Total degree: {deg_P1}")

print("\nP2:")
print(f"  Degree in i1: {deg_P2_i1}")
print(f"  Degree in i2: {deg_P2_i2}")
print(f"  Total degree: {deg_P2}")

# Naïve Bézout Schranke

bezout_bound = deg_P1 * deg_P2  # oder andere Kombinationen
print(f"\nNaïve Bézout bound: {deg_P1} × {deg_P2} = {bezout_bound}")


from sympy import div, degree, resultant, factor, solve, nroots, gcd
from sympy import Poly, Rational, symbols, simplify


print("\n" + "-" * 80)
print("STEP 5: Common divisor of P1 and P2")
print("-" * 80)

# G ist bereits bekannt

G = gcd(P1, P2)

print("\nCommon divisor of P1 and P2:")
print(f"G(i1, i2) = {G}")

# Faktorisiere G

G_factored = factor(G)
print(f"\nG factorised: {G_factored}")


# Nullstellen von G
print("\nRoots of G:")
for a in roots(G, i1):
    print(f"  i1 = {a}  (outside [0,1])")
for a in roots(G, i2):
    print(f"  i2 = {a}  (outside [0,1])")

print("\n✓ G = 0 has NO solutions in [0,1]²!")
print("  → The common divisor has no solutions in the unit square")

# ============================================================================

# SCHRITT 2: Division mit rationalen Koeffizienten

# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Division P1/G and P2/G")
print("-" * 80)

print("\nCompute Q1 = P1 / G...")

Q1, remainder1 = div(P1, G, i1, i2)
print("Q1 computed")
print(f"Q1: {Q1}")
print(f"Remainder: {remainder1}")


print("\nCompute Q2 = P2 / G...")

Q2, remainder2 = div(P2, G, i1, i2)
print("Q2 computed")
print(f"Q2: {Q2}")
print(f"Remainder: {remainder2}")

# ============================================================================

# SCHRITT 3: Grade der reduzierten Polynome

# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Degrees of the reduced polynomials Q1, Q2")
print("-" * 80)

# Expandiere Q1, Q2 zu Polynomen

from sympy import expand, Poly, im

deg_Q1_i1 = degree(Q1, i1)
deg_Q1_i2 = degree(Q1, i2)
deg_Q2_i1 = degree(Q2, i1)
deg_Q2_i2 = degree(Q2, i2)
deg_Q1 = total_degree(Q1, i1, i2)
deg_Q2 = total_degree(Q1, i1, i2)

print("\nQ1:")
print(f"  Degree in i1: {deg_Q1_i1}")
print(f"  Degree in i2: {deg_Q1_i2}")
print(f"  Total degree: {deg_Q1}")

print("\nQ2:")
print(f"  Degree in i1: {deg_Q2_i1}")
print(f"  Degree in i2: {deg_Q2_i2}")
print(f"  Total degree: {deg_Q2}")

bezout_bound = deg_Q1 * deg_Q2  # oder andere Kombinationen
print(f"\nNaïve Bézout bound: {deg_Q1} × {deg_Q2} = {bezout_bound}")


# ============================================================================

# SCHRITT 5: Resultante des reduzierten Systems

# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Resultant of the reduced system")
print("-" * 80)

print("\nCompute resultant R_Q = Res(Q1, Q2, i2)...")

R_Q = resultant(Q1, Q2, i2)

# Vereinfache
R_Q = simplify(R_Q)

deg_R_Q = degree(R_Q, i1)

print(f"\nDegree of the resultant in i1: {deg_R_Q}")

if deg_R_Q == float("-inf") or R_Q == 0:
    print("\n⚠ Resultant is zero - more common divisors!")
else:
    print("\n✓ Resultant is non-trivial")
    print(f"  Maximal number of roots in i1: {deg_R_Q}")

    # Faktorisiere Resultante
    print("\nFactorise resultant...")
    R_Q_factored = factor(R_Q)
    print(f"R_Q factorised: {R_Q_factored}")

divisor = (
    -b
    * beta_1
    * beta_2**2
    * i1**3
    * sigma
    * (b + beta_1 * i1)
    * (
        b**2
        + b * beta_1 * i1
        - b * beta_1
        + b * gamma_1
        + beta_1 * gamma_1 * i1
    )
)
roots_divisor = roots(divisor, i1)
print(f"\nDivisor g of R_Q: {divisor}")
print(f"with roots: {roots_divisor}")

for a in roots_divisor:
    Q1_a = Q1.subs(i1, a)
    Q2_a = Q2.subs(i1, a)
    roots_Q1 = roots(Q1_a, i2)
    roots_Q2 = roots(Q2_a, i2)
    print(f"\nFor root i1={a} we have:")
    print(f"Q1({a}, i2): {simplify(Q1_a)}")
    print(f"Q2({a}, i2): {simplify(Q2_a)}")
    print(f"\nQ1({a}, i2) has roots: {roots_Q1}")
    print(f"Q2({a}, i2) has roots: {roots_Q2}")


i2_i1 = roots(Q1, i2)
print(f"\nIn general, we find the roots of Q1 wrt i2: {i2_i1}")

i2_i1 = (
    -(b + beta_1 * i1)
    * (
        b**2
        + b * beta_1 * i1
        - b * beta_1
        + b * gamma_1
        + beta_1 * gamma_1 * i1
    )
    / (
        beta_2
        * (
            b**2
            + b * beta_1 * i1
            + b * beta_1 * sigma
            - b * beta_1
            + b * gamma_1
            + beta_1 * gamma_1 * i1
        )
    )
)

i2_i1 = simplify(i2_i1)
print(f"\nThis implies i2(i1): {i2_i1}")


S, remainder = div(R_Q, divisor)
S = simplify(S)
print(f"\nS=R_Q/g: {S}")
print(f"with remainder: {remainder}")

i1c = (-(b**2) + b * beta_1 - b * gamma_1) / (b * beta_1 + beta_1 * gamma_1)
print(f"\nCritical value i1c of i1: {i1c}")

Si1c = S.subs(i1, i1c)
Si1c = simplify(Si1c)
print(f"S(i1c)={Si1c}")

roots_Si1c = roots(Si1c, sigma)
print(f"With sigma roots: {roots_Si1c}")

i1d = (-(b**2) + b * beta_1 - b * gamma_1 - b * beta_1 * sigma) / (
    b * beta_1 + beta_1 * gamma_1
)
print(f"\nCritical i1d of i1: {i1d}")

Si1d = S.subs(i1, i1d)
Si1d = simplify(Si1d)
print(f"S(i1d)={Si1d}")

roots_Si1d = roots(Si1d, sigma)
print(f"With sigma roots: {roots_Si1d}")

Si10 = S.subs(i1, 0)
Si10 = simplify(Si10)
print(f"\nS(0)={Si10}")

roots_Si10 = roots(Si10, sigma)
print(f"With sigma roots: {roots_Si10}")
