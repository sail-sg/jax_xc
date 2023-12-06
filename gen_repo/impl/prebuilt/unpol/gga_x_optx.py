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
t21 = params_a_gamma ** 2
t23 = s0 ** 2
t25 = 2 ** (0.1e1 / 0.3e1)
t26 = r0 ** 2
t27 = t26 ** 2
t33 = t25 ** 2
t34 = t19 ** 2
t40 = (0.1e1 + params_a_gamma * s0 * t33 / t34 / t26) ** 2
t49 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params_a_a + 0.2e1 * params_a_b * t21 * t23 * t25 / t19 / t27 / r0 / t40))
res = 0.2e1 * t49
