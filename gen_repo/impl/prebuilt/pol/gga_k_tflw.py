t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = lax_cond(t15, t16, t18)
t20 = lax_cond(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = lax_cond(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t34 = r0 ** 2
t35 = r0 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t39 = 6 ** (0.1e1 / 0.3e1)
t41 = math.pi ** 2
t42 = t41 ** (0.1e1 / 0.3e1)
t43 = t42 ** 2
t44 = 0.1e1 / t43
t52 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (params_a_gamma + 0.5e1 / 0.72e2 * params_a_lambda * s0 / t36 / t34 * t39 * t44))
t54 = lax_cond(t11, t16, -t18)
t55 = lax_cond(t15, t12, t54)
t56 = 0.1e1 + t55
t58 = t56 ** (0.1e1 / 0.3e1)
t59 = t58 ** 2
t61 = lax_cond(t56 <= p_a_zeta_threshold, t25, t59 * t56)
t64 = r1 ** 2
t65 = r1 ** (0.1e1 / 0.3e1)
t66 = t65 ** 2
t77 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t61 * t31 * (params_a_gamma + 0.5e1 / 0.72e2 * params_a_lambda * s2 / t66 / t64 * t39 * t44))
res = t52 + t77
