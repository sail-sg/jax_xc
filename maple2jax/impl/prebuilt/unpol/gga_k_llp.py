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
t27 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t29 = 4 ** (0.1e1 / 0.3e1)
t32 = 2 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = r0 ** 2
t39 = math.sqrt(s0)
t42 = 0.1e1 / t22 / r0
t46 = math.asinh(t39 * t32 * t42)
t59 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.10e1 + 0.2e1 / 0.9e1 * params_a_beta * t4 / t27 * t29 * s0 * t33 / t23 / t35 / (0.10e1 + params_a_gamma * params_a_beta * t39 * t32 * t42 * t46)))
res = 0.2e1 * t59
