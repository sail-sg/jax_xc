t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t22 = 2 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t24 = r0 ** 2
t25 = t19 ** 2
t28 = t23 / t25 / t24
t32 = (params_a_gamma * s0 * t28 + 0.1e1) ** params_a_omega
t40 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params_a_beta * s0 * t28 / t32))
res = 0.2e1 * t40
