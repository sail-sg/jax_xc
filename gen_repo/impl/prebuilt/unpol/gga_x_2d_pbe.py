t3 = math.sqrt(math.pi)
t5 = 0.1e1 <= p_a_zeta_threshold
t6 = p_a_zeta_threshold - 0.1e1
t8 = lax_cond(t5, -t6, 0)
t9 = lax_cond(t5, t6, t8)
t10 = 0.1e1 + t9
t12 = math.sqrt(p_a_zeta_threshold)
t14 = math.sqrt(t10)
t16 = lax_cond(t10 <= p_a_zeta_threshold, t12 * p_a_zeta_threshold, t14 * t10)
t18 = math.sqrt(0.2e1)
t19 = math.sqrt(r0)
t23 = r0 ** 2
t35 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t19 * (0.14604e1 - 0.21196816e0 / (0.4604e0 + 0.44318359375000000000e-1 / math.pi * s0 / t23 / r0)))
res = 0.2e1 * t35
