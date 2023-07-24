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
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = 2 ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t24 = t20 ** 2
t29 = r0 ** 2
t34 = tau0 * t22 / t24 / r0 - s0 * t22 / t24 / t29 / 0.8e1
t35 = 6 ** (0.1e1 / 0.3e1)
t37 = math.pi ** 2
t38 = t37 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t45 = t34 ** 2
t47 = t35 ** 2
t49 = 0.1e1 / t38 / t37
t54 = (0.1e1 + 0.25e2 / 0.81e2 * params_a_e1 * t45 * t47 * t49) ** 2
t55 = t45 ** 2
t57 = t37 ** 2
t64 = (t54 + 0.1250e4 / 0.2187e4 * params_a_c1 * t55 * t35 / t39 / t57) ** (0.1e1 / 0.4e1)
t71 = s0 ** 2
t73 = t29 ** 2
t81 = (0.1e1 + params_a_b * t47 * t49 * t71 * t21 / t20 / t73 / r0 / 0.288e3) ** (0.1e1 / 0.8e1)
t86 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + params_a_k0 * (0.1e1 - 0.5e1 / 0.9e1 * t34 * t35 / t39) / t64) / t81)
res = 0.2e1 * t86
