t1 = math.sqrt(0.2e1)
t2 = math.sqrt(math.pi)
t6 = r0 + r1
t8 = (r0 - r1) / t6
t9 = 0.1e1 + t8
t11 = math.sqrt(p_a_zeta_threshold)
t12 = t11 * p_a_zeta_threshold
t13 = math.sqrt(t9)
t15 = lax_cond(t9 <= p_a_zeta_threshold, t12, t13 * t9)
t16 = 0.1e1 - t8
t18 = math.sqrt(t16)
t20 = lax_cond(t16 <= p_a_zeta_threshold, t12, t18 * t16)
t23 = math.sqrt(t6)
res = -0.4e1 / 0.3e1 * t1 / t2 * (t15 / 0.2e1 + t20 / 0.2e1) * t23
