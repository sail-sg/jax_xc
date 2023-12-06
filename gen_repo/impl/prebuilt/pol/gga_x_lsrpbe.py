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
t29 = 6 ** (0.1e1 / 0.3e1)
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t35 = params_a_mu * t29 * t34
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t40 = 0.1e1 / t38 / t36
t42 = 0.1e1 / params_a_kappa
t46 = math.exp(-t35 * s0 * t40 * t42 / 0.24e2)
t49 = params_a_kappa + 0.1e1
t50 = params_a_alpha * t29
t55 = math.exp(-t50 * t34 * s0 * t40 / 0.24e2)
t62 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params_a_kappa * (0.1e1 - t46) - t49 * (0.1e1 - t55)))
t64 = lax_cond(t10, t15, -t17)
t65 = lax_cond(t14, t11, t64)
t66 = 0.1e1 + t65
t68 = t66 ** (0.1e1 / 0.3e1)
t70 = lax_cond(t66 <= p_a_zeta_threshold, t23, t68 * t66)
t72 = r1 ** 2
t73 = r1 ** (0.1e1 / 0.3e1)
t74 = t73 ** 2
t76 = 0.1e1 / t74 / t72
t81 = math.exp(-t35 * s2 * t76 * t42 / 0.24e2)
t88 = math.exp(-t50 * t34 * s2 * t76 / 0.24e2)
t95 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t70 * t27 * (0.1e1 + params_a_kappa * (0.1e1 - t81) - t49 * (0.1e1 - t88)))
res = t62 + t95
