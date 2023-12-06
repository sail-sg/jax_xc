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
t23 = r0 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = r0 ** 2
t33 = t29 / t24 / t30
t36 = s0 ** 2
t38 = t30 ** 2
t42 = t28 / t23 / t38 / r0
t46 = t36 * s0
t48 = t38 ** 2
t49 = 0.1e1 / t48
t53 = t36 ** 2
t58 = t29 / t24 / t48 / t30
t84 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t24 * (params_a_aa[1] * s0 * t33 + 0.2e1 * params_a_aa[2] * t36 * t42 + 0.4e1 * params_a_aa[3] * t46 * t49 + 0.4e1 * params_a_aa[4] * t53 * t58 + params_a_aa[0]) / (params_a_bb[1] * s0 * t33 + 0.2e1 * params_a_bb[2] * t36 * t42 + 0.4e1 * params_a_bb[3] * t46 * t49 + 0.4e1 * params_a_bb[4] * t53 * t58 + params_a_bb[0]))
res = 0.2e1 * t84
