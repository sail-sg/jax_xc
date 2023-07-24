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
t31 = t7 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t33 = params_a_aa[0]
t34 = params_a_aa[1]
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t40 = 0.1e1 / t38 / t36
t42 = params_a_aa[2]
t43 = s0 ** 2
t45 = t36 ** 2
t48 = 0.1e1 / t37 / t45 / r0
t50 = params_a_aa[3]
t51 = t43 * s0
t53 = t45 ** 2
t54 = 0.1e1 / t53
t56 = params_a_aa[4]
t57 = t43 ** 2
t61 = 0.1e1 / t38 / t53 / t36
t65 = params_a_bb[0]
t66 = params_a_bb[1]
t69 = params_a_bb[2]
t72 = params_a_bb[3]
t75 = params_a_bb[4]
t83 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t32 * (t34 * s0 * t40 + t42 * t43 * t48 + t50 * t51 * t54 + t56 * t57 * t61 + t33) / (t66 * s0 * t40 + t69 * t43 * t48 + t72 * t51 * t54 + t75 * t57 * t61 + t65))
t85 = lax_cond(t11, t16, -t18)
t86 = lax_cond(t15, t12, t85)
t87 = 0.1e1 + t86
t89 = t87 ** (0.1e1 / 0.3e1)
t90 = t89 ** 2
t92 = lax_cond(t87 <= p_a_zeta_threshold, t25, t90 * t87)
t95 = r1 ** 2
t96 = r1 ** (0.1e1 / 0.3e1)
t97 = t96 ** 2
t99 = 0.1e1 / t97 / t95
t101 = s2 ** 2
t103 = t95 ** 2
t106 = 0.1e1 / t96 / t103 / r1
t108 = t101 * s2
t110 = t103 ** 2
t111 = 0.1e1 / t110
t113 = t101 ** 2
t117 = 0.1e1 / t97 / t110 / t95
t134 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t92 * t32 * (t34 * s2 * t99 + t42 * t101 * t106 + t50 * t108 * t111 + t56 * t113 * t117 + t33) / (t66 * s2 * t99 + t69 * t101 * t106 + t72 * t108 * t111 + t75 * t113 * t117 + t65))
res = t83 + t134
