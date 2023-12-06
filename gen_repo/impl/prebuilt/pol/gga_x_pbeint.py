t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = lax_cond(t14, t15, t17)
t19 = lax_cond(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t31 = 6 ** (0.1e1 / 0.3e1)
t32 = (params_a_muPBE - params_a_muGE) * params_a_alpha * t31
t33 = math.pi ** 2
t34 = t33 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t36 = 0.1e1 / t35
t37 = t36 * s0
t38 = r0 ** 2
t39 = r0 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
t42 = 0.1e1 / t40 / t38
t43 = params_a_alpha * t31
t44 = t37 * t42
t66 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + (params_a_muGE + t32 * t37 * t42 / (0.1e1 + t43 * t44 / 0.24e2) / 0.24e2) * t31 * t44 / 0.24e2))))
t68 = lax_cond(t10, t15, -t17)
t69 = lax_cond(t14, t11, t68)
t70 = 0.1e1 + t69
t72 = t70 ** (0.1e1 / 0.3e1)
t74 = lax_cond(t70 <= p_a_zeta_threshold, t23, t72 * t70)
t76 = t36 * s2
t77 = r1 ** 2
t78 = r1 ** (0.1e1 / 0.3e1)
t79 = t78 ** 2
t81 = 0.1e1 / t79 / t77
t82 = t76 * t81
t104 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t74 * t27 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + (params_a_muGE + t32 * t76 * t81 / (0.1e1 + t43 * t82 / 0.24e2) / 0.24e2) * t31 * t82 / 0.24e2))))
res = t66 + t104
