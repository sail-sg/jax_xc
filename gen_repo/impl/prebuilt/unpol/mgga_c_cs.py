t1 = r0 ** (0.1e1 / 0.3e1)
t2 = 0.1e1 / t1
t7 = math.exp(-0.25330000000000000000e0 * t2)
t9 = p_a_zeta_threshold ** 2
t10 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t11 = t10 ** 2
t13 = lax_cond(0.1e1 <= p_a_zeta_threshold, t11 * t9, 1)
t14 = 2 ** (0.1e1 / 0.3e1)
t16 = t14 ** 2
t18 = t1 ** 2
t20 = 0.1e1 / t18 / r0
t28 = r0 ** 2
res = -0.4918e-1 / (0.1e1 + 0.34899999999999999998e0 * t2) * (0.1e1 + 0.264e0 * t7 * (t13 * t14 * (tau0 * t16 * t20 - l0 * t16 * t20 / 0.8e1) / 0.4e1 - s0 / t18 / t28 / 0.8e1 + l0 * t20 / 0.8e1))
