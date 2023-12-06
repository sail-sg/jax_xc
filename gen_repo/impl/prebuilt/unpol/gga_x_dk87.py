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
t21 = 0.1e1 / math.pi
t22 = 6 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = math.pi ** 2
t26 = t25 ** (0.1e1 / 0.3e1)
t28 = t3 ** 2
t30 = t21 ** (0.1e1 / 0.3e1)
t34 = 4 ** (0.1e1 / 0.3e1)
t36 = 2 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t39 = r0 ** 2
t40 = t19 ** 2
t42 = 0.1e1 / t40 / t39
t43 = math.sqrt(s0)
t48 = (t43 * t36 / t19 / r0) ** params_a_alpha
t65 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.7e1 / 0.11664e5 * t21 * t23 / t26 * t28 / t30 * t34 * s0 * t37 * t42 * (params_a_a1 * t48 + 0.1e1) / (params_a_b1 * s0 * t37 * t42 + 0.1e1)))
res = 0.2e1 * t65
