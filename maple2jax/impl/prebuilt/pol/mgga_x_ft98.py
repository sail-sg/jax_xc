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
t37 = math.sqrt(params_a_a1 * s0 * t34 + 0.1e1)
t42 = (params_a_b1 * s0 * t34 + 0.1e1) ** (0.1e1 / 0.4e1)
t43 = t42 ** 2
t49 = s0 * t34
t54 = (t49 - l0 / t32 / r0) ** 2
t57 = (0.1e1 + t49) ** 2
t62 = params_a_b2 ** 2
t64 = math.sqrt(t62 + 0.1e1)
t65 = t64 - params_a_b2
t66 = s0 ** 2
t67 = t30 ** 2
t71 = t66 / t31 / t67 / r0
t72 = l0 ** 2
t76 = t72 / t31 / t30 / r0
t77 = t71 - t76 - params_a_b2
t78 = DBL_EPSILON ** (0.1e1 / 0.4e1)
t79 = 0.1e1 / t78
t83 = 0.2e1 * params_a_b2
t89 = lax_cond(0.0e0 < t77, t77, -t77)
t91 = t77 ** 2
t93 = t91 ** 2
t97 = lax_cond(-t79 < t77, t77, -t79)
t98 = t97 ** 2
t100 = math.sqrt(0.1e1 + t98)
t103 = lax_cond(t89 < t78, 0.1e1 - t71 + t76 + params_a_b2 + t91 / 0.2e1 - t93 / 0.8e1, 0.1e1 / (t97 + t100))
t104 = lax_cond(t77 < -t79, -0.2e1 * t71 + 0.2e1 * t76 + t83 - 0.1e1 / t77 / 0.2e1, t103)
t107 = 2 ** (0.1e1 / 0.3e1)
t109 = (t107 - 0.1e1) * t65
t111 = t104 * t109 + 0.1e1
t112 = t111 ** 2
t119 = t2 ** 2
t121 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t122 = t121 ** 2
t124 = 4 ** (0.1e1 / 0.3e1)
t125 = t119 * t122 * t124
t133 = math.sqrt((0.1e1 + params_a_a * t37 / t43 / t42 * s0 * t34 + params_a_b * (0.1e1 + params_a_a2 * t54 / t57) * (t104 * t65 + 0.1e1) / t112 / t111 * t54) / (0.1e1 + 0.81e2 / 0.4e1 * t125 * params_a_b * s0 * t34))
t137 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t133)
t139 = lax_cond(t10, t15, -t17)
t140 = lax_cond(t14, t11, t139)
t141 = 0.1e1 + t140
t143 = t141 ** (0.1e1 / 0.3e1)
t145 = lax_cond(t141 <= p_a_zeta_threshold, t23, t143 * t141)
t148 = r1 ** 2
t149 = r1 ** (0.1e1 / 0.3e1)
t150 = t149 ** 2
t152 = 0.1e1 / t150 / t148
t155 = math.sqrt(s2 * t152 * params_a_a1 + 0.1e1)
t160 = (s2 * t152 * params_a_b1 + 0.1e1) ** (0.1e1 / 0.4e1)
t161 = t160 ** 2
t167 = s2 * t152
t172 = (t167 - l1 / t150 / r1) ** 2
t175 = (0.1e1 + t167) ** 2
t180 = s2 ** 2
t181 = t148 ** 2
t185 = t180 / t149 / t181 / r1
t186 = l1 ** 2
t190 = t186 / t149 / t148 / r1
t191 = t185 - t190 - params_a_b2
t200 = lax_cond(0.0e0 < t191, t191, -t191)
t202 = t191 ** 2
t204 = t202 ** 2
t208 = lax_cond(-t79 < t191, t191, -t79)
t209 = t208 ** 2
t211 = math.sqrt(0.1e1 + t209)
t214 = lax_cond(t200 < t78, 0.1e1 - t185 + t190 + params_a_b2 + t202 / 0.2e1 - t204 / 0.8e1, 0.1e1 / (t208 + t211))
t215 = lax_cond(t191 < -t79, -0.2e1 * t185 + 0.2e1 * t190 + t83 - 0.1e1 / t191 / 0.2e1, t214)
t219 = t109 * t215 + 0.1e1
t220 = t219 ** 2
t234 = math.sqrt((0.1e1 + params_a_a * t155 / t161 / t160 * s2 * t152 + params_a_b * (0.1e1 + params_a_a2 * t172 / t175) * (t215 * t65 + 0.1e1) / t220 / t219 * t172) / (0.1e1 + 0.81e2 / 0.4e1 * t125 * params_a_b * s2 * t152))
t238 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t145 * t27 * t234)
res = t137 + t238
