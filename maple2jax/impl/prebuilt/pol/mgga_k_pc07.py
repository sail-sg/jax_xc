t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = lax_cond(t15, t16, t18)
t20 = lax_cond(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = lax_cond(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = 6 ** (0.1e1 / 0.3e1)
t34 = math.pi ** 2
t35 = t34 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = t33 / t36
t39 = r0 ** 2
t40 = r0 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t45 = t38 * s0 / t41 / t39
t46 = 0.5e1 / 0.72e2 * t45
t53 = t33 ** 2
t56 = t53 / t35 / t34
t57 = l0 ** 2
t63 = t56 * t57 / t40 / t39 / r0 / 0.5832e4
t64 = t39 ** 2
t70 = t56 * s0 / t40 / t64 * l0 / 0.5184e4
t71 = s0 ** 2
t77 = t56 * t71 / t40 / t64 / r0 / 0.17496e5
t80 = (t63 - t70 + t77) ** 2
t82 = (0.1e1 + t46) ** 2
t86 = math.sqrt(0.1e1 + t80 / t82)
t89 = (0.1e1 + 0.5e1 / 0.648e3 * t45 + 0.5e1 / 0.54e2 * t38 * l0 / t41 / r0 + t63 - t70 + t77) / t86 - t46
t90 = params_a_a / 0.40e2
t92 = 0.39e2 / 0.40e2 * params_a_a
t94 = params_a_a * params_a_b
t96 = lax_cond(t89 < t90, t90, t89)
t98 = lax_cond(t96 < t92, t96, t92)
t99 = 0.1e1 / t98
t101 = math.exp(-t94 * t99)
t105 = math.exp(-params_a_a / (params_a_a - t98))
t107 = (0.1e1 + t105) ** params_a_b
t110 = math.exp(-params_a_a * t99)
t112 = (t110 + t105) ** params_a_b
t115 = lax_cond(t92 <= t89, 1, t101 * t107 / t112)
t116 = lax_cond(t89 <= t90, 0, t115)
t122 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (t89 * t116 + t46))
t124 = lax_cond(t11, t16, -t18)
t125 = lax_cond(t15, t12, t124)
t126 = 0.1e1 + t125
t128 = t126 ** (0.1e1 / 0.3e1)
t129 = t128 ** 2
t131 = lax_cond(t126 <= p_a_zeta_threshold, t25, t129 * t126)
t133 = r1 ** 2
t134 = r1 ** (0.1e1 / 0.3e1)
t135 = t134 ** 2
t139 = t38 * s2 / t135 / t133
t140 = 0.5e1 / 0.72e2 * t139
t147 = l1 ** 2
t153 = t56 * t147 / t134 / t133 / r1 / 0.5832e4
t154 = t133 ** 2
t160 = t56 * s2 / t134 / t154 * l1 / 0.5184e4
t161 = s2 ** 2
t167 = t56 * t161 / t134 / t154 / r1 / 0.17496e5
t170 = (t153 - t160 + t167) ** 2
t172 = (0.1e1 + t140) ** 2
t176 = math.sqrt(0.1e1 + t170 / t172)
t179 = (0.1e1 + 0.5e1 / 0.648e3 * t139 + 0.5e1 / 0.54e2 * t38 * l1 / t135 / r1 + t153 - t160 + t167) / t176 - t140
t183 = lax_cond(t179 < t90, t90, t179)
t185 = lax_cond(t183 < t92, t183, t92)
t186 = 0.1e1 / t185
t188 = math.exp(-t94 * t186)
t192 = math.exp(-params_a_a / (params_a_a - t185))
t194 = (0.1e1 + t192) ** params_a_b
t197 = math.exp(-params_a_a * t186)
t199 = (t197 + t192) ** params_a_b
t202 = lax_cond(t92 <= t179, 1, t188 * t194 / t199)
t203 = lax_cond(t179 <= t90, 0, t202)
t209 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t131 * t31 * (t179 * t203 + t140))
res = t122 + t209
