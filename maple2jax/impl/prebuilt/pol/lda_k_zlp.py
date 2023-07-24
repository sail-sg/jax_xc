t1 = 3 ** (0.1e1 / 0.3e1)
t2 = t1 ** 2
t4 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t7 = 4 ** (0.1e1 / 0.3e1)
t10 = r0 + r1
t12 = (r0 - r1) / t10
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t17 = t16 * p_a_zeta_threshold
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = lax_cond(t13 <= p_a_zeta_threshold, t17, t19 * t13)
t22 = 0.1e1 - t12
t24 = t22 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t27 = lax_cond(t22 <= p_a_zeta_threshold, t17, t25 * t22)
t30 = t10 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t36 = math.log(0.1e1 + 0.51020408163265306120e3 / t30)
res = 0.10790666666666666667e1 * t2 / t4 * t7 * (t21 / 0.2e1 + t27 / 0.2e1) * t31 * (0.1e1 - 0.19600000000000000000e-2 * t30 * t36)
