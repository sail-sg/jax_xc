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
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t40 = s0 / t37 / t35
t42 = 0.5e1 / 0.972e3 * t34 * t40
t47 = params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t42))
t50 = tau0 / t37 / r0
t52 = t50 - t40 / 0.8e1
t53 = t52 ** 2
t54 = t29 ** 2
t56 = 0.3e1 / 0.10e2 * t54 * t32
t57 = t50 + t56
t58 = t57 ** 2
t62 = 0.1e1 - 0.4e1 * t53 / t58
t63 = t62 ** 2
t70 = t53 ** 2
t73 = t58 ** 2
t92 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t47 + t63 * t62 / (0.1e1 + 0.8e1 * t53 * t52 / t58 / t57 + 0.64e2 * params_a_b * t70 * t53 / t73 / t58) * (params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t42 + params_a_c)) - t47)))
t94 = lax_cond(t10, t15, -t17)
t95 = lax_cond(t14, t11, t94)
t96 = 0.1e1 + t95
t98 = t96 ** (0.1e1 / 0.3e1)
t100 = lax_cond(t96 <= p_a_zeta_threshold, t23, t98 * t96)
t102 = r1 ** 2
t103 = r1 ** (0.1e1 / 0.3e1)
t104 = t103 ** 2
t107 = s2 / t104 / t102
t109 = 0.5e1 / 0.972e3 * t34 * t107
t114 = params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t109))
t117 = tau1 / t104 / r1
t119 = t117 - t107 / 0.8e1
t120 = t119 ** 2
t121 = t117 + t56
t122 = t121 ** 2
t126 = 0.1e1 - 0.4e1 * t120 / t122
t127 = t126 ** 2
t134 = t120 ** 2
t137 = t122 ** 2
t156 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t100 * t27 * (0.1e1 + t114 + t127 * t126 / (0.1e1 + 0.8e1 * t120 * t119 / t122 / t121 + 0.64e2 * params_a_b * t134 * t120 / t137 / t122) * (params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t109 + params_a_c)) - t114)))
res = t92 + t156
