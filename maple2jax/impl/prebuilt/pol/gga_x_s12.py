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
t28 = t6 ** (0.1e1 / 0.3e1)
t29 = t28 * params_a_bx
t31 = r0 ** 2
t32 = r0 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = 0.1e1 / t33 / t31
t37 = s0 ** 2
t39 = t31 ** 2
t58 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t29 * (params_a_A + params_a_B * (0.1e1 - 0.1e1 / (0.1e1 + params_a_C * s0 * t35 + params_a_D * t37 / t32 / t39 / r0)) * (0.1e1 - 0.1e1 / (params_a_E * s0 * t35 + 0.1e1))))
t60 = lax_cond(t10, t15, -t17)
t61 = lax_cond(t14, t11, t60)
t62 = 0.1e1 + t61
t64 = t62 ** (0.1e1 / 0.3e1)
t66 = lax_cond(t62 <= p_a_zeta_threshold, t23, t64 * t62)
t69 = r1 ** 2
t70 = r1 ** (0.1e1 / 0.3e1)
t71 = t70 ** 2
t73 = 0.1e1 / t71 / t69
t75 = s2 ** 2
t77 = t69 ** 2
t96 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t66 * t29 * (params_a_A + params_a_B * (0.1e1 - 0.1e1 / (0.1e1 + params_a_C * s2 * t73 + params_a_D * t75 / t70 / t77 / r1)) * (0.1e1 - 0.1e1 / (params_a_E * s2 * t73 + 0.1e1))))
res = t58 + t96
