t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = t3 ** 2
t24 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t26 = 4 ** (0.1e1 / 0.3e1)
t28 = 6 ** (0.1e1 / 0.3e1)
t29 = math.pi ** 2
t30 = t29 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t34 = 2 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t37 = r0 ** 2
t38 = t20 ** 2
t47 = 0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t28 / t31 * s0 * t35 / t38 / t37)
t51 = math.sqrt(math.pi * t21 / t24 * t26 / t47)
t55 = (t12 * r0) ** (0.1e1 / 0.3e1)
t59 = p_a_cam_omega / t51 * t34 / t55 / 0.2e1
t61 = 0.192e1 < t59
t62 = lax_cond(t61, t59, 0.192e1)
t63 = t62 ** 2
t64 = t63 ** 2
t67 = t64 * t63
t70 = t64 ** 2
t73 = t70 * t63
t76 = t70 * t64
t79 = t70 * t67
t82 = t70 ** 2
t106 = t82 ** 2
t117 = -0.1e1 / t64 / 0.30e2 + 0.1e1 / t67 / 0.70e2 - 0.1e1 / t70 / 0.135e3 + 0.1e1 / t73 / 0.231e3 - 0.1e1 / t76 / 0.364e3 + 0.1e1 / t79 / 0.540e3 - 0.1e1 / t82 / 0.765e3 + 0.1e1 / t82 / t63 / 0.1045e4 - 0.1e1 / t82 / t64 / 0.1386e4 + 0.1e1 / t82 / t67 / 0.1794e4 - 0.1e1 / t82 / t70 / 0.2275e4 + 0.1e1 / t82 / t73 / 0.2835e4 - 0.1e1 / t82 / t76 / 0.3480e4 + 0.1e1 / t82 / t79 / 0.4216e4 - 0.1e1 / t106 / 0.5049e4 + 0.1e1 / t106 / t63 / 0.5985e4 - 0.1e1 / t106 / t64 / 0.7030e4 + 0.1e1 / t63 / 0.9e1
t118 = lax_cond(t61, 0.192e1, t59)
t119 = math.atan2(0.1e1, t118)
t120 = t118 ** 2
t124 = math.log(0.1e1 + 0.1e1 / t120)
t133 = lax_cond(0.192e1 <= t59, t117, 0.1e1 - 0.8e1 / 0.3e1 * t118 * (t119 + t118 * (0.1e1 - (t120 + 0.3e1) * t124) / 0.4e1))
t138 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t133 * t47)
res = 0.2e1 * t138
