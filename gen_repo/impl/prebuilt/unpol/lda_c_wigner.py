t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 ** (0.1e1 / 0.3e1)
res = params_a_a / (params_a_b + t1 * t3 * t6 / t7 / 0.4e1)
