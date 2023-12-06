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
t29 = params_a_CC_0_[0]
t30 = params_a_CC_0_[1]
t32 = r0 ** 2
t33 = r0 ** (0.1e1 / 0.3e1)
t34 = t33 ** 2
t36 = 0.1e1 / t34 / t32
t39 = 0.1e1 + 0.4e-2 * s0 * t36
t41 = t36 / t39
t44 = params_a_CC_0_[2]
t45 = s0 ** 2
t47 = t32 ** 2
t51 = t39 ** 2
t53 = 0.1e1 / t33 / t47 / r0 / t51
t56 = params_a_CC_0_[3]
t57 = t45 * s0
t59 = t47 ** 2
t63 = 0.1e1 / t59 / t51 / t39
t66 = params_a_CC_1_[0]
t67 = params_a_CC_1_[1]
t71 = params_a_CC_1_[2]
t75 = params_a_CC_1_[3]
t81 = 2 ** (0.1e1 / 0.3e1)
t82 = 0.1e1 / t27 * t81
t84 = 0.1e1 + t17 <= p_a_zeta_threshold
t86 = 0.1e1 - t17 <= p_a_zeta_threshold
t87 = lax_cond(t86, t15, t17)
t88 = lax_cond(t84, t11, t87)
t89 = 0.1e1 + t88
t91 = 0.1e1 / t22
t92 = t89 ** (0.1e1 / 0.3e1)
t94 = lax_cond(t89 <= p_a_zeta_threshold, t91, 0.1e1 / t92)
t97 = 0.1e1 + 0.39999999999999999998e0 * t82 * t94
t100 = params_a_CC_2_[0]
t101 = params_a_CC_2_[1]
t105 = params_a_CC_2_[2]
t109 = params_a_CC_2_[3]
t114 = t97 ** 2
t117 = params_a_CC_3_[0]
t118 = params_a_CC_3_[1]
t122 = params_a_CC_3_[2]
t126 = params_a_CC_3_[3]
t138 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t29 + 0.4e-2 * t30 * s0 * t41 + 0.16e-4 * t44 * t45 * t53 + 0.64e-7 * t56 * t57 * t63 + (t66 + 0.4e-2 * t67 * s0 * t41 + 0.16e-4 * t71 * t45 * t53 + 0.64e-7 * t75 * t57 * t63) / t97 + (t100 + 0.4e-2 * t101 * s0 * t41 + 0.16e-4 * t105 * t45 * t53 + 0.64e-7 * t109 * t57 * t63) / t114 + (t117 + 0.4e-2 * t118 * s0 * t41 + 0.16e-4 * t122 * t45 * t53 + 0.64e-7 * t126 * t57 * t63) / t114 / t97))
t140 = lax_cond(t10, t15, -t17)
t141 = lax_cond(t14, t11, t140)
t142 = 0.1e1 + t141
t144 = t142 ** (0.1e1 / 0.3e1)
t146 = lax_cond(t142 <= p_a_zeta_threshold, t23, t144 * t142)
t149 = r1 ** 2
t150 = r1 ** (0.1e1 / 0.3e1)
t151 = t150 ** 2
t153 = 0.1e1 / t151 / t149
t156 = 0.1e1 + 0.4e-2 * s2 * t153
t158 = t153 / t156
t161 = s2 ** 2
t163 = t149 ** 2
t167 = t156 ** 2
t169 = 0.1e1 / t150 / t163 / r1 / t167
t172 = t161 * s2
t174 = t163 ** 2
t178 = 0.1e1 / t174 / t167 / t156
t191 = lax_cond(t84, t15, -t17)
t192 = lax_cond(t86, t11, t191)
t193 = 0.1e1 + t192
t195 = t193 ** (0.1e1 / 0.3e1)
t197 = lax_cond(t193 <= p_a_zeta_threshold, t91, 0.1e1 / t195)
t200 = 0.1e1 + 0.39999999999999999998e0 * t82 * t197
t213 = t200 ** 2
t233 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t146 * t27 * (t29 + 0.4e-2 * t30 * s2 * t158 + 0.16e-4 * t44 * t161 * t169 + 0.64e-7 * t56 * t172 * t178 + (t66 + 0.4e-2 * t67 * s2 * t158 + 0.16e-4 * t71 * t161 * t169 + 0.64e-7 * t75 * t172 * t178) / t200 + (t100 + 0.4e-2 * t101 * s2 * t158 + 0.16e-4 * t105 * t161 * t169 + 0.64e-7 * t109 * t172 * t178) / t213 + (t117 + 0.4e-2 * t118 * s2 * t158 + 0.16e-4 * t122 * t161 * t169 + 0.64e-7 * t126 * t172 * t178) / t213 / t200))
res = t138 + t233
