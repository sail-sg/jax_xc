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
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t2 ** 2
t32 = 4 ** (0.1e1 / 0.3e1)
t34 = 0.8e1 / 0.27e2 * t29 * t30 * t32
t35 = r0 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t40 = r0 ** 2
t43 = s0 / t36 / t40
t46 = 6 ** (0.1e1 / 0.3e1)
t48 = math.pi ** 2
t49 = t48 ** (0.1e1 / 0.3e1)
t50 = t49 ** 2
t51 = 0.1e1 / t50
t52 = (tau0 / t36 / r0 - t43 / 0.8e1) * t46 * t51
t59 = 0.1e1 - t34
t64 = 0.5e1 / 0.9e1 * t52
t65 = 0.1e1 - t64
t66 = Heaviside(t65)
t74 = Heaviside(-t65)
t84 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((t34 + 0.5e1 / 0.9e1 * t52 * (0.827411e0 - 0.35753333333333333333e0 * t52) / (0.10e1 - 0.45341611111111111111e0 * t52) * t59) * t66 + (0.1e1 + 0.148e0 * t65 / (0.1e1 + t64)) * t74) / (0.1e1 + 0.1015549e-2 * t43))
t86 = lax_cond(t10, t15, -t17)
t87 = lax_cond(t14, t11, t86)
t88 = 0.1e1 + t87
t90 = t88 ** (0.1e1 / 0.3e1)
t92 = lax_cond(t88 <= p_a_zeta_threshold, t23, t90 * t88)
t94 = r1 ** (0.1e1 / 0.3e1)
t95 = t94 ** 2
t99 = r1 ** 2
t102 = s2 / t95 / t99
t106 = (tau1 / t95 / r1 - t102 / 0.8e1) * t46 * t51
t117 = 0.5e1 / 0.9e1 * t106
t118 = 0.1e1 - t117
t119 = Heaviside(t118)
t127 = Heaviside(-t118)
t137 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t92 * t28 * ((t34 + 0.5e1 / 0.9e1 * t106 * (0.827411e0 - 0.35753333333333333333e0 * t106) / (0.10e1 - 0.45341611111111111111e0 * t106) * t59) * t119 + (0.1e1 + 0.148e0 * t118 / (0.1e1 + t117)) * t127) / (0.1e1 + 0.1015549e-2 * t102))
res = t84 + t137
