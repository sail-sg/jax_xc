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
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t33 = 0.1e1 / t32
t34 = t29 * t33
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t39 = 0.1e1 / t37 / t35
t40 = s0 * t39
t45 = 0.100e3 / 0.6561e4 / params_a_k1 - 0.73e2 / 0.648e3
t46 = t29 ** 2
t50 = t45 * t46 / t31 / t30
t51 = s0 ** 2
t52 = t35 ** 2
t57 = t45 * t29
t59 = t33 * s0 * t39
t62 = math.exp(-0.27e2 / 0.80e2 * t57 * t59)
t66 = math.sqrt(0.146e3)
t67 = t66 * t29
t77 = 0.5e1 / 0.9e1 * (tau0 / t37 / r0 - t40 / 0.8e1) * t29 * t33
t78 = 0.1e1 - t77
t80 = t78 ** 2
t82 = math.exp(-t80 / 0.2e1)
t86 = (0.7e1 / 0.12960e5 * t67 * t59 + t66 * t78 * t82 / 0.100e3) ** 2
t94 = math.log(DBL_EPSILON)
t97 = t94 / (-t94 + params_a_c1)
t100 = lax_cond(t77 < -t97, t77, -t97)
t105 = math.exp(-params_a_c1 * t100 / (0.1e1 - t100))
t106 = lax_cond(-t97 < t77, 0, t105)
t107 = abs(params_a_d)
t110 = math.log(DBL_EPSILON / t107)
t113 = (-t110 + params_a_c2) / t110
t114 = t77 < -t113
t115 = lax_cond(t114, -t113, t77)
t119 = math.exp(params_a_c2 / (0.1e1 - t115))
t121 = lax_cond(t114, 0, -params_a_d * t119)
t122 = lax_cond(t77 <= 0.1e1, t106, t121)
t128 = math.sqrt(0.3e1)
t130 = t46 / t31
t131 = math.sqrt(s0)
t136 = math.sqrt(t130 * t131 / t36 / r0)
t140 = math.exp(-0.98958000000000000000e1 * t128 / t136)
t145 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((0.1e1 + params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + 0.5e1 / 0.972e3 * t34 * t40 + t50 * t51 / t36 / t52 / r0 * t62 / 0.576e3 + t86))) * (0.1e1 - t122) + 0.1174e1 * t122) * (0.1e1 - t140))
t147 = lax_cond(t10, t15, -t17)
t148 = lax_cond(t14, t11, t147)
t149 = 0.1e1 + t148
t151 = t149 ** (0.1e1 / 0.3e1)
t153 = lax_cond(t149 <= p_a_zeta_threshold, t23, t151 * t149)
t155 = r1 ** 2
t156 = r1 ** (0.1e1 / 0.3e1)
t157 = t156 ** 2
t159 = 0.1e1 / t157 / t155
t160 = s2 * t159
t163 = s2 ** 2
t164 = t155 ** 2
t170 = t33 * s2 * t159
t173 = math.exp(-0.27e2 / 0.80e2 * t57 * t170)
t186 = 0.5e1 / 0.9e1 * (tau1 / t157 / r1 - t160 / 0.8e1) * t29 * t33
t187 = 0.1e1 - t186
t189 = t187 ** 2
t191 = math.exp(-t189 / 0.2e1)
t195 = (0.7e1 / 0.12960e5 * t67 * t170 + t66 * t187 * t191 / 0.100e3) ** 2
t205 = lax_cond(t186 < -t97, t186, -t97)
t210 = math.exp(-params_a_c1 * t205 / (0.1e1 - t205))
t211 = lax_cond(-t97 < t186, 0, t210)
t212 = t186 < -t113
t213 = lax_cond(t212, -t113, t186)
t217 = math.exp(params_a_c2 / (0.1e1 - t213))
t219 = lax_cond(t212, 0, -params_a_d * t217)
t220 = lax_cond(t186 <= 0.1e1, t211, t219)
t226 = math.sqrt(s2)
t231 = math.sqrt(t130 * t226 / t156 / r1)
t235 = math.exp(-0.98958000000000000000e1 * t128 / t231)
t240 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t153 * t28 * ((0.1e1 + params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + 0.5e1 / 0.972e3 * t34 * t160 + t50 * t163 / t156 / t164 / r1 * t173 / 0.576e3 + t195))) * (0.1e1 - t220) + 0.1174e1 * t220) * (0.1e1 - t235))
res = t145 + t240
