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
t30 = params_a_mu * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t35 = t30 * t34
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t39 = t38 * t36
t40 = 0.1e1 / t39
t42 = params_a_alpha * t29
t44 = t34 * s0 * t40
t47 = math.exp(-t42 * t44 / 0.24e2)
t56 = t29 ** 2
t57 = params_a_alpha * t56
t59 = 0.1e1 / t32 / t31
t60 = s0 ** 2
t62 = t36 ** 2
t69 = math.exp(-t57 * t59 * t60 / t37 / t62 / r0 / 0.576e3)
t81 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t35 * s0 * t40 * t47 / (0.1e1 + t30 * t44 / 0.24e2) / 0.24e2 + 0.4e1 * (0.1e1 - t69) * t56 * t33 / s0 * t39 + t69))
t83 = lax_cond(t10, t15, -t17)
t84 = lax_cond(t14, t11, t83)
t85 = 0.1e1 + t84
t87 = t85 ** (0.1e1 / 0.3e1)
t89 = lax_cond(t85 <= p_a_zeta_threshold, t23, t87 * t85)
t91 = r1 ** 2
t92 = r1 ** (0.1e1 / 0.3e1)
t93 = t92 ** 2
t94 = t93 * t91
t95 = 0.1e1 / t94
t98 = t34 * s2 * t95
t101 = math.exp(-t42 * t98 / 0.24e2)
t110 = s2 ** 2
t112 = t91 ** 2
t119 = math.exp(-t57 * t59 * t110 / t92 / t112 / r1 / 0.576e3)
t131 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t89 * t27 * (t35 * s2 * t95 * t101 / (0.1e1 + t30 * t98 / 0.24e2) / 0.24e2 + 0.4e1 * (0.1e1 - t119) * t56 * t33 / s2 * t94 + t119))
res = t81 + t131
