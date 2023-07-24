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
t29 = r0 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t33 = tau0 / t30 / r0
t34 = r0 ** 2
t39 = t33 - s0 / t30 / t34 / 0.8e1
t40 = 6 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t42 = math.pi ** 2
t43 = t42 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t46 = 0.3e1 / 0.10e2 * t41 * t44
t47 = t33 - t46
t52 = t39 ** 2
t54 = t47 ** 2
t58 = (0.1e1 + params_a_e1 * t52 / t54) ** 2
t59 = t52 ** 2
t61 = t54 ** 2
t65 = (t58 + params_a_c1 * t59 / t61) ** (0.1e1 / 0.4e1)
t70 = params_a_b * t41
t72 = 0.1e1 / t43 / t42
t73 = s0 ** 2
t75 = t34 ** 2
t83 = (0.1e1 + t70 * t72 * t73 / t29 / t75 / r0 / 0.576e3) ** (0.1e1 / 0.8e1)
t88 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + params_a_k0 * (0.1e1 - t39 / t47) / t65) / t83)
t90 = lax_cond(t10, t15, -t17)
t91 = lax_cond(t14, t11, t90)
t92 = 0.1e1 + t91
t94 = t92 ** (0.1e1 / 0.3e1)
t96 = lax_cond(t92 <= p_a_zeta_threshold, t23, t94 * t92)
t98 = r1 ** (0.1e1 / 0.3e1)
t99 = t98 ** 2
t102 = tau1 / t99 / r1
t103 = r1 ** 2
t108 = t102 - s2 / t99 / t103 / 0.8e1
t109 = t102 - t46
t114 = t108 ** 2
t116 = t109 ** 2
t120 = (0.1e1 + params_a_e1 * t114 / t116) ** 2
t121 = t114 ** 2
t123 = t116 ** 2
t127 = (t120 + params_a_c1 * t121 / t123) ** (0.1e1 / 0.4e1)
t132 = s2 ** 2
t134 = t103 ** 2
t142 = (0.1e1 + t70 * t72 * t132 / t98 / t134 / r1 / 0.576e3) ** (0.1e1 / 0.8e1)
t147 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t96 * t28 * (0.1e1 + params_a_k0 * (0.1e1 - t108 / t109) / t127) / t142)
res = t88 + t147
