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
t22 = r0 ** 2
t25 = math.sqrt(s0)
t26 = t25 * t18
t28 = 0.1e1 / t19 / r0
t30 = math.asinh(t26 * t28)
t43 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t19 * (0.1e1 + 0.52500000000000000000e-2 * t3 * s0 / t22 / r0 / (0.1e1 + 0.56e-1 * t26 * t28 * t30)))
res = 0.2e1 * t43
