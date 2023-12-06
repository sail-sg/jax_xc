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
t29 = t2 ** 2
t30 = math.pi * t29
t32 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t34 = 4 ** (0.1e1 / 0.3e1)
t35 = 0.1e1 / t32 * t34
t36 = 6 ** (0.1e1 / 0.3e1)
t37 = params_a_mu * t36
t38 = math.pi ** 2
t39 = t38 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
t41 = 0.1e1 / t40
t43 = r0 ** 2
t44 = r0 ** (0.1e1 / 0.3e1)
t45 = t44 ** 2
t56 = 0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t37 * t41 * s0 / t45 / t43 / 0.24e2))
t60 = math.sqrt(t30 * t35 / t56)
t63 = 2 ** (0.1e1 / 0.3e1)
t65 = (t20 * t6) ** (0.1e1 / 0.3e1)
t69 = p_a_cam_omega / t60 * t63 / t65 / 0.2e1
t71 = 0.135e1 < t69
t72 = lax_cond(t71, t69, 0.135e1)
t73 = t72 ** 2
t76 = t73 ** 2
t79 = t76 * t73
t82 = t76 ** 2
t94 = t82 ** 2
t98 = lax_cond(t71, 0.135e1, t69)
t99 = math.sqrt(math.pi)
t102 = math.erf(0.1e1 / t98 / 0.2e1)
t104 = t98 ** 2
t107 = math.exp(-0.1e1 / t104 / 0.4e1)
t118 = lax_cond(0.135e1 <= t69, 0.1e1 / t73 / 0.36e2 - 0.1e1 / t76 / 0.960e3 + 0.1e1 / t79 / 0.26880e5 - 0.1e1 / t82 / 0.829440e6 + 0.1e1 / t82 / t73 / 0.28385280e8 - 0.1e1 / t82 / t76 / 0.1073479680e10 + 0.1e1 / t82 / t79 / 0.44590694400e11 - 0.1e1 / t94 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t98 * (t99 * t102 + 0.2e1 * t98 * (t107 - 0.3e1 / 0.2e1 - 0.2e1 * t104 * (t107 - 0.1e1))))
t123 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t118 * t56)
t125 = lax_cond(t10, t15, -t17)
t126 = lax_cond(t14, t11, t125)
t127 = 0.1e1 + t126
t129 = t127 ** (0.1e1 / 0.3e1)
t131 = lax_cond(t127 <= p_a_zeta_threshold, t23, t129 * t127)
t134 = r1 ** 2
t135 = r1 ** (0.1e1 / 0.3e1)
t136 = t135 ** 2
t147 = 0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t37 * t41 * s2 / t136 / t134 / 0.24e2))
t151 = math.sqrt(t30 * t35 / t147)
t155 = (t127 * t6) ** (0.1e1 / 0.3e1)
t159 = p_a_cam_omega / t151 * t63 / t155 / 0.2e1
t161 = 0.135e1 < t159
t162 = lax_cond(t161, t159, 0.135e1)
t163 = t162 ** 2
t166 = t163 ** 2
t169 = t166 * t163
t172 = t166 ** 2
t184 = t172 ** 2
t188 = lax_cond(t161, 0.135e1, t159)
t191 = math.erf(0.1e1 / t188 / 0.2e1)
t193 = t188 ** 2
t196 = math.exp(-0.1e1 / t193 / 0.4e1)
t207 = lax_cond(0.135e1 <= t159, 0.1e1 / t163 / 0.36e2 - 0.1e1 / t166 / 0.960e3 + 0.1e1 / t169 / 0.26880e5 - 0.1e1 / t172 / 0.829440e6 + 0.1e1 / t172 / t163 / 0.28385280e8 - 0.1e1 / t172 / t166 / 0.1073479680e10 + 0.1e1 / t172 / t169 / 0.44590694400e11 - 0.1e1 / t184 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t188 * (t99 * t191 + 0.2e1 * t188 * (t196 - 0.3e1 / 0.2e1 - 0.2e1 * t193 * (t196 - 0.1e1))))
t212 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t131 * t28 * t207 * t147)
res = t123 + t212
