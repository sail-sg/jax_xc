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
t30 = r0 ** 2
t31 = r0 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = 0.1e1 / t32 / t30
t35 = 2 ** (0.1e1 / 0.3e1)
t38 = t20 ** 2
t39 = t6 ** 2
t40 = t38 * t39
t42 = (t20 * t6) ** (0.1e1 / 0.3e1)
t43 = t42 ** 2
t44 = s0 * t34
t55 = params_a_beta0 + params_a_beta1 * s0 * t34 * t35 * t40 * t43 / (params_a_beta2 + t44 * t35 * t40 * t43 / 0.8e1) / 0.8e1
t58 = t2 ** 2
t60 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t62 = t58 / t60
t63 = 4 ** (0.1e1 / 0.3e1)
t64 = t55 ** 2
t65 = math.asinh(t44)
t66 = t65 ** 2
t71 = math.sqrt(0.9e1 * t44 * t64 * t66 + 0.1e1)
t81 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t55 * s0 * t34 * t62 * t63 / t71))
t83 = lax_cond(t10, t15, -t17)
t84 = lax_cond(t14, t11, t83)
t85 = 0.1e1 + t84
t87 = t85 ** (0.1e1 / 0.3e1)
t89 = lax_cond(t85 <= p_a_zeta_threshold, t23, t87 * t85)
t92 = r1 ** 2
t93 = r1 ** (0.1e1 / 0.3e1)
t94 = t93 ** 2
t96 = 0.1e1 / t94 / t92
t99 = t85 ** 2
t100 = t99 * t39
t102 = (t85 * t6) ** (0.1e1 / 0.3e1)
t103 = t102 ** 2
t104 = s2 * t96
t115 = params_a_beta0 + params_a_beta1 * s2 * t96 * t35 * t100 * t103 / (params_a_beta2 + t104 * t35 * t100 * t103 / 0.8e1) / 0.8e1
t118 = t115 ** 2
t119 = math.asinh(t104)
t120 = t119 ** 2
t125 = math.sqrt(0.9e1 * t104 * t118 * t120 + 0.1e1)
t135 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t89 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t115 * s2 * t96 * t62 * t63 / t125))
res = t81 + t135
