t3 = r0 + r1
t4 = t3 ** 2
t5 = t3 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t11 = r0 ** (0.1e1 / 0.3e1)
t12 = t11 ** 2
t18 = (r0 - r1) / t3
t20 = 0.1e1 / 0.2e1 + t18 / 0.2e1
t21 = t20 ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t26 = r1 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t32 = 0.1e1 / 0.2e1 - t18 / 0.2e1
t33 = t32 ** (0.1e1 / 0.3e1)
t34 = t33 ** 2
res = -(0.80569e0 + 0.37655000000000000000e-3 * (s0 + 0.2e1 * s1 + s2) / t6 / t4 - 0.37655000000000000000e-3 * l0 / t12 / r0 * t22 * t20 - 0.37655000000000000000e-3 * l1 / t27 / r1 * t34 * t32) / (0.1e1 / t5 + 0.40743e-2)
