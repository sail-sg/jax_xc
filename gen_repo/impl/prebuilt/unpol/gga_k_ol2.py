t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = lax_cond(t8, -t9, 0)
t12 = lax_cond(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = lax_cond(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t26 = 2 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t28 = r0 ** 2
t34 = math.sqrt(s0)
t37 = 0.1e1 / t22 / r0
t50 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (params_a_aa + 0.13888888888888888889e-1 * params_a_bb * s0 * t27 / t23 / t28 + params_a_cc * t34 * t26 * t37 / (0.4e1 * t34 * t26 * t37 + t26)))
res = 0.2e1 * t50
