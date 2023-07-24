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
t45 = t34 ** 2
t46 = 0.1e1 / t45
t47 = s0 ** 2
t50 = t39 ** 2
t51 = t50 ** 2
t64 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + 0.5e1 / 0.648e3 * t38 * s0 / t41 / t39 / (0.1e1 + t46 * t47 * s0 / t51 / 0.2304e4)))
t66 = lax_cond(t11, t16, -t18)
t67 = lax_cond(t15, t12, t66)
t68 = 0.1e1 + t67
t70 = t68 ** (0.1e1 / 0.3e1)
t71 = t70 ** 2
t73 = lax_cond(t68 <= p_a_zeta_threshold, t25, t71 * t68)
t75 = r1 ** 2
t76 = r1 ** (0.1e1 / 0.3e1)
t77 = t76 ** 2
t81 = s2 ** 2
t84 = t75 ** 2
t85 = t84 ** 2
t98 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t73 * t31 * (0.1e1 + 0.5e1 / 0.648e3 * t38 * s2 / t77 / t75 / (0.1e1 + t46 * t81 * s2 / t85 / 0.2304e4)))
res = t64 + t98
