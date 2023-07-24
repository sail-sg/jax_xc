t3 = math.pi ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = 0.1e1 <= p_a_zeta_threshold
t6 = p_a_zeta_threshold - 0.1e1
t8 = lax_cond(t5, -t6, 0)
t9 = lax_cond(t5, t6, t8)
t10 = 0.1e1 + t9
t12 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t14 = t10 ** (0.1e1 / 0.3e1)
t16 = lax_cond(t10 <= p_a_zeta_threshold, t12 * p_a_zeta_threshold, t14 * t10)
t18 = r0 ** 2
t19 = 0.1e1 / tau0
t22 = 2 ** (0.1e1 / 0.3e1)
t30 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t32 = 4 ** (0.1e1 / 0.3e1)
t37 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.27e2 / 0.160e3 * t4 * t16 * t18 * t19 * t22 * (0.1e1 + 0.7e1 / 0.216e3 * s0 / r0 * t19) / t30 * t32)
res = 0.2e1 * t37
