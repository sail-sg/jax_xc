t2 = r0 + r1
t4 = (r0 - r1) / t2
t6 = lax_cond(0.0e0 < t4, t4, -t4)
t8 = lax_cond(0.1e-9 < t6, t6, 0.1e-9)
t9 = t8 ** (0.1e1 / 0.3e1)
t10 = t9 ** 2
t13 = math.sqrt(-t10 * t8 + 0.1e1)
t15 = s0 + 0.2e1 * s1 + s2
t16 = math.sqrt(t15)
t18 = t2 ** 2
t19 = t18 ** 2
t22 = t2 ** (0.1e1 / 0.3e1)
t26 = (t16 / t22 / t2) ** (0.1e1 / 0.16e2)
t27 = t26 ** 2
t35 = 3 ** (0.1e1 / 0.3e1)
t37 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t39 = 4 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
res = -t13 / (0.118e2 + 0.150670e0 * t27 * t26 * t16 * t15 / t19 + 0.11020000000000000000e-1 * t15 / t18 / t2 + t35 * t37 * t40 / t22 / 0.4e1)
