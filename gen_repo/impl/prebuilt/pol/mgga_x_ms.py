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
t52 = tau0 / t37 / r0 - t40 / 0.8e1
t53 = t52 ** 2
t54 = t29 ** 2
t57 = 0.1e1 / t31 / t30
t60 = 0.1e1 - 0.25e2 / 0.81e2 * t53 * t54 * t57
t61 = t60 ** 2
t64 = t30 ** 2
t65 = 0.1e1 / t64
t68 = t53 ** 2
t71 = t64 ** 2
t72 = 0.1e1 / t71
t89 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t47 + t61 * t60 / (0.1e1 + 0.250e3 / 0.243e3 * t53 * t52 * t65 + 0.62500e5 / 0.59049e5 * params_a_b * t68 * t53 * t72) * (params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t42 + params_a_c)) - t47)))
t91 = lax_cond(t10, t15, -t17)
t92 = lax_cond(t14, t11, t91)
t93 = 0.1e1 + t92
t95 = t93 ** (0.1e1 / 0.3e1)
t97 = lax_cond(t93 <= p_a_zeta_threshold, t23, t95 * t93)
t99 = r1 ** 2
t100 = r1 ** (0.1e1 / 0.3e1)
t101 = t100 ** 2
t104 = s2 / t101 / t99
t106 = 0.5e1 / 0.972e3 * t34 * t104
t111 = params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t106))
t116 = tau1 / t101 / r1 - t104 / 0.8e1
t117 = t116 ** 2
t121 = 0.1e1 - 0.25e2 / 0.81e2 * t117 * t54 * t57
t122 = t121 ** 2
t127 = t117 ** 2
t146 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t97 * t27 * (0.1e1 + t111 + t122 * t121 / (0.1e1 + 0.250e3 / 0.243e3 * t117 * t116 * t65 + 0.62500e5 / 0.59049e5 * params_a_b * t127 * t117 * t72) * (params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t106 + params_a_c)) - t111)))
res = t89 + t146
