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
t33 = 0.1e1 / t32
t34 = 4 ** (0.1e1 / 0.3e1)
t35 = t33 * t34
t37 = t29 * t33 * t34
t38 = r0 ** 2
t39 = r0 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
t44 = math.sqrt(s0)
t47 = t44 / t39 / r0
t48 = math.asinh(t47)
t56 = 0.1e1 + 0.93333333333333333332e-3 * t37 * s0 / t40 / t38 / (0.1e1 + 0.2520e-1 * t47 * t48)
t60 = math.sqrt(t30 * t35 / t56)
t63 = 2 ** (0.1e1 / 0.3e1)
t65 = (t20 * t6) ** (0.1e1 / 0.3e1)
t69 = p_a_cam_omega / t60 * t63 / t65 / 0.2e1
t71 = 0.192e1 < t69
t72 = lax_cond(t71, t69, 0.192e1)
t73 = t72 ** 2
t74 = t73 ** 2
t77 = t74 * t73
t80 = t74 ** 2
t83 = t80 * t73
t86 = t80 * t74
t89 = t80 * t77
t92 = t80 ** 2
t116 = t92 ** 2
t127 = -0.1e1 / t74 / 0.30e2 + 0.1e1 / t77 / 0.70e2 - 0.1e1 / t80 / 0.135e3 + 0.1e1 / t83 / 0.231e3 - 0.1e1 / t86 / 0.364e3 + 0.1e1 / t89 / 0.540e3 - 0.1e1 / t92 / 0.765e3 + 0.1e1 / t92 / t73 / 0.1045e4 - 0.1e1 / t92 / t74 / 0.1386e4 + 0.1e1 / t92 / t77 / 0.1794e4 - 0.1e1 / t92 / t80 / 0.2275e4 + 0.1e1 / t92 / t83 / 0.2835e4 - 0.1e1 / t92 / t86 / 0.3480e4 + 0.1e1 / t92 / t89 / 0.4216e4 - 0.1e1 / t116 / 0.5049e4 + 0.1e1 / t116 / t73 / 0.5985e4 - 0.1e1 / t116 / t74 / 0.7030e4 + 0.1e1 / t73 / 0.9e1
t128 = lax_cond(t71, 0.192e1, t69)
t129 = math.atan2(0.1e1, t128)
t130 = t128 ** 2
t134 = math.log(0.1e1 + 0.1e1 / t130)
t143 = lax_cond(0.192e1 <= t69, t127, 0.1e1 - 0.8e1 / 0.3e1 * t128 * (t129 + t128 * (0.1e1 - (t130 + 0.3e1) * t134) / 0.4e1))
t148 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * t143 * t56)
t150 = lax_cond(t10, t15, -t17)
t151 = lax_cond(t14, t11, t150)
t152 = 0.1e1 + t151
t154 = t152 ** (0.1e1 / 0.3e1)
t156 = lax_cond(t152 <= p_a_zeta_threshold, t23, t154 * t152)
t158 = r1 ** 2
t159 = r1 ** (0.1e1 / 0.3e1)
t160 = t159 ** 2
t164 = math.sqrt(s2)
t167 = t164 / t159 / r1
t168 = math.asinh(t167)
t176 = 0.1e1 + 0.93333333333333333332e-3 * t37 * s2 / t160 / t158 / (0.1e1 + 0.2520e-1 * t167 * t168)
t180 = math.sqrt(t30 * t35 / t176)
t184 = (t152 * t6) ** (0.1e1 / 0.3e1)
t188 = p_a_cam_omega / t180 * t63 / t184 / 0.2e1
t190 = 0.192e1 < t188
t191 = lax_cond(t190, t188, 0.192e1)
t192 = t191 ** 2
t193 = t192 ** 2
t196 = t193 * t192
t199 = t193 ** 2
t202 = t199 * t192
t205 = t199 * t193
t208 = t199 * t196
t211 = t199 ** 2
t235 = t211 ** 2
t246 = -0.1e1 / t193 / 0.30e2 + 0.1e1 / t196 / 0.70e2 - 0.1e1 / t199 / 0.135e3 + 0.1e1 / t202 / 0.231e3 - 0.1e1 / t205 / 0.364e3 + 0.1e1 / t208 / 0.540e3 - 0.1e1 / t211 / 0.765e3 + 0.1e1 / t211 / t192 / 0.1045e4 - 0.1e1 / t211 / t193 / 0.1386e4 + 0.1e1 / t211 / t196 / 0.1794e4 - 0.1e1 / t211 / t199 / 0.2275e4 + 0.1e1 / t211 / t202 / 0.2835e4 - 0.1e1 / t211 / t205 / 0.3480e4 + 0.1e1 / t211 / t208 / 0.4216e4 - 0.1e1 / t235 / 0.5049e4 + 0.1e1 / t235 / t192 / 0.5985e4 - 0.1e1 / t235 / t193 / 0.7030e4 + 0.1e1 / t192 / 0.9e1
t247 = lax_cond(t190, 0.192e1, t188)
t248 = math.atan2(0.1e1, t247)
t249 = t247 ** 2
t253 = math.log(0.1e1 + 0.1e1 / t249)
t262 = lax_cond(0.192e1 <= t188, t246, 0.1e1 - 0.8e1 / 0.3e1 * t247 * (t248 + t247 * (0.1e1 - (t249 + 0.3e1) * t253) / 0.4e1))
t267 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t156 * t28 * t262 * t176)
res = t148 + t267
