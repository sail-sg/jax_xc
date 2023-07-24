t2 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t5 = lax_cond(0.1e1 <= p_a_zeta_threshold, t3 * p_a_zeta_threshold, 1)
t7 = 3 ** (0.1e1 / 0.3e1)
t10 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t11 = t10 ** 2
t13 = 4 ** (0.1e1 / 0.3e1)
t14 = t13 ** 2
t16 = r0 ** (0.1e1 / 0.3e1)
t17 = t16 ** 2
res = params_a_ax * t5 * t7 / t11 * t14 * t17 / 0.3e1
