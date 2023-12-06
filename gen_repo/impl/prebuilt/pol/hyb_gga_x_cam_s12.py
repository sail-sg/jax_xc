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
t30 = r0 ** 2
t31 = r0 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = 0.1e1 / t32 / t30
t36 = s0 ** 2
t38 = t30 ** 2
t53 = params_a_A + params_a_B * (0.1e1 - 0.1e1 / (0.1e1 + params_a_C * s0 * t34 + params_a_D * t36 / t31 / t38 / r0)) * (0.1e1 - 0.1e1 / (params_a_E * s0 * t34 + 0.1e1))
t55 = t2 ** 2
t56 = math.pi * t55
t58 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t60 = 4 ** (0.1e1 / 0.3e1)
t61 = 0.1e1 / t58 * t60
t65 = math.sqrt(t56 * t61 / t53)
t68 = 2 ** (0.1e1 / 0.3e1)
t70 = (t20 * t6) ** (0.1e1 / 0.3e1)
t74 = p_a_cam_omega / t65 * t68 / t70 / 0.2e1
t76 = 0.135e1 < t74
t77 = lax_cond(t76, t74, 0.135e1)
t78 = t77 ** 2
t81 = t78 ** 2
t84 = t81 * t78
t87 = t81 ** 2
t99 = t87 ** 2
t103 = lax_cond(t76, 0.135e1, t74)
t104 = math.sqrt(math.pi)
t107 = math.erf(0.1e1 / t103 / 0.2e1)
t109 = t103 ** 2
t112 = math.exp(-0.1e1 / t109 / 0.4e1)
t123 = lax_cond(0.135e1 <= t74, 0.1e1 / t78 / 0.36e2 - 0.1e1 / t81 / 0.960e3 + 0.1e1 / t84 / 0.26880e5 - 0.1e1 / t87 / 0.829440e6 + 0.1e1 / t87 / t78 / 0.28385280e8 - 0.1e1 / t87 / t81 / 0.1073479680e10 + 0.1e1 / t87 / t84 / 0.44590694400e11 - 0.1e1 / t99 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t103 * (t104 * t107 + 0.2e1 * t103 * (t112 - 0.3e1 / 0.2e1 - 0.2e1 * t109 * (t112 - 0.1e1))))
t129 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t53 * (-p_a_cam_beta * t123 - p_a_cam_alpha + 0.1e1))
t131 = lax_cond(t10, t15, -t17)
t132 = lax_cond(t14, t11, t131)
t133 = 0.1e1 + t132
t135 = t133 ** (0.1e1 / 0.3e1)
t137 = lax_cond(t133 <= p_a_zeta_threshold, t23, t135 * t133)
t140 = r1 ** 2
t141 = r1 ** (0.1e1 / 0.3e1)
t142 = t141 ** 2
t144 = 0.1e1 / t142 / t140
t146 = s2 ** 2
t148 = t140 ** 2
t163 = params_a_A + params_a_B * (0.1e1 - 0.1e1 / (0.1e1 + params_a_C * s2 * t144 + params_a_D * t146 / t141 / t148 / r1)) * (0.1e1 - 0.1e1 / (params_a_E * s2 * t144 + 0.1e1))
t168 = math.sqrt(t56 * t61 / t163)
t172 = (t133 * t6) ** (0.1e1 / 0.3e1)
t176 = p_a_cam_omega / t168 * t68 / t172 / 0.2e1
t178 = 0.135e1 < t176
t179 = lax_cond(t178, t176, 0.135e1)
t180 = t179 ** 2
t183 = t180 ** 2
t186 = t183 * t180
t189 = t183 ** 2
t201 = t189 ** 2
t205 = lax_cond(t178, 0.135e1, t176)
t208 = math.erf(0.1e1 / t205 / 0.2e1)
t210 = t205 ** 2
t213 = math.exp(-0.1e1 / t210 / 0.4e1)
t224 = lax_cond(0.135e1 <= t176, 0.1e1 / t180 / 0.36e2 - 0.1e1 / t183 / 0.960e3 + 0.1e1 / t186 / 0.26880e5 - 0.1e1 / t189 / 0.829440e6 + 0.1e1 / t189 / t180 / 0.28385280e8 - 0.1e1 / t189 / t183 / 0.1073479680e10 + 0.1e1 / t189 / t186 / 0.44590694400e11 - 0.1e1 / t201 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t205 * (t104 * t208 + 0.2e1 * t205 * (t213 - 0.3e1 / 0.2e1 - 0.2e1 * t210 * (t213 - 0.1e1))))
t230 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t137 * t28 * t163 * (-p_a_cam_beta * t224 - p_a_cam_alpha + 0.1e1))
res = t129 + t230
