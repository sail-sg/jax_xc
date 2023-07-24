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
t30 = params_a_alpha * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t41 = t34 * s0 / t38 / t36
t44 = math.exp(-t30 * t41 / 0.24e2)
t50 = t29 ** 2
t51 = 0.1e1 / t32
t52 = t50 * t51
t53 = math.sqrt(s0)
t55 = 0.1e1 / t37 / r0
t59 = (t52 * t53 * t55 / 0.12e2) ** params_a_expo
t60 = params_a_f * t59
t64 = params_a_b * t50
t69 = math.asinh(t64 * t51 * t53 * t55 / 0.12e2)
t80 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + ((params_a_d * t44 + params_a_c) * t29 * t41 / 0.24e2 - t60) / (0.1e1 + t52 * t53 * t55 * params_a_a * t69 / 0.12e2 + t60)))
t82 = lax_cond(t10, t15, -t17)
t83 = lax_cond(t14, t11, t82)
t84 = 0.1e1 + t83
t86 = t84 ** (0.1e1 / 0.3e1)
t88 = lax_cond(t84 <= p_a_zeta_threshold, t23, t86 * t84)
t91 = r1 ** 2
t92 = r1 ** (0.1e1 / 0.3e1)
t93 = t92 ** 2
t96 = t34 * s2 / t93 / t91
t99 = math.exp(-t30 * t96 / 0.24e2)
t105 = math.sqrt(s2)
t107 = 0.1e1 / t92 / r1
t111 = (t52 * t105 * t107 / 0.12e2) ** params_a_expo
t112 = params_a_f * t111
t120 = math.asinh(t64 * t51 * t105 * t107 / 0.12e2)
t131 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t88 * t27 * (0.1e1 + ((params_a_d * t99 + params_a_c) * t29 * t96 / 0.24e2 - t112) / (0.1e1 + t52 * t105 * t107 * params_a_a * t120 / 0.12e2 + t112)))
res = t80 + t131
