t2 = r0 ** 2
t3 = r0 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t6 = 0.1e1 / t4 / t2
t9 = math.exp(-params_a_k * s0 * t6)
t13 = 3 ** (0.1e1 / 0.3e1)
t15 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t17 = 4 ** (0.1e1 / 0.3e1)
t18 = t17 ** 2
t22 = t13 ** 2
t23 = math.pi ** (0.1e1 / 0.3e1)
t25 = math.sqrt(s0)
t27 = t2 ** 2
t33 = math.sqrt(t25 / t3 / r0)
res = (params_a_b * s0 * t6 * t9 + params_a_a) / (params_a_c + t13 * t15 * t18 / t3 * (0.1e1 + params_a_d * t17 * t22 * t23 * t33 * t25 * s0 / t27 / 0.3e1) / 0.4e1)
