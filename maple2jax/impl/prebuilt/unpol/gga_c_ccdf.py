t1 = r0 ** (0.1e1 / 0.3e1)
t7 = 2 ** (0.1e1 / 0.3e1)
t8 = 6 ** (0.1e1 / 0.3e1)
t9 = t8 ** 2
t11 = math.pi ** 2
t12 = t11 ** (0.1e1 / 0.3e1)
t14 = math.sqrt(s0)
t23 = math.exp(-params_a_c4 * (t7 * t9 / t12 * t14 / t1 / r0 / 0.12e2 - params_a_c5))
res = params_a_c1 / (0.1e1 + params_a_c2 / t1) * (0.1e1 - params_a_c3 / (0.1e1 + t23))
