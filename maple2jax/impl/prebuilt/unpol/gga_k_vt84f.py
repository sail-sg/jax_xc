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
t25 = 6 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t27 = math.pi ** 2
t28 = t27 ** (0.1e1 / 0.3e1)
t31 = math.sqrt(s0)
t32 = 2 ** (0.1e1 / 0.3e1)
t38 = t26 / t28 * t31 * t32 / t22 / r0 / 0.12e2
t39 = math.sqrt(DBL_EPSILON)
t43 = t28 ** 2
t46 = t32 ** 2
t48 = r0 ** 2
t55 = params_a_mu ** 2
t61 = s0 ** 2
t63 = t48 ** 2
t72 = lax_cond(t39 < t38, t38, t39)
t73 = t72 ** 2
t74 = params_a_mu * t73
t76 = math.exp(-params_a_alpha * t73)
t81 = t73 ** 2
t83 = math.exp(-params_a_alpha * t81)
t90 = lax_cond(t38 <= t39, 0.1e1 + (-params_a_mu + params_a_alpha + 0.5e1 / 0.3e1) * t25 / t43 * s0 * t46 / t23 / t48 / 0.24e2 + (params_a_mu * params_a_alpha + t55 - params_a_alpha) * t26 / t28 / t27 * t61 * t32 / t22 / t63 / r0 / 0.288e3, 0.1e1 - t74 * t76 / (0.1e1 + t74) + (0.1e1 - t83) * (0.1e1 / t73 - 0.1e1) + 0.5e1 / 0.3e1 * t73)
t94 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * t90)
res = 0.2e1 * t94
