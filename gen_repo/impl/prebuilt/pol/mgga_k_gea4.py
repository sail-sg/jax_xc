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
t52 = t33 ** 2
t55 = t52 / t35 / t34
t56 = l0 ** 2
t63 = t39 ** 2
t70 = s0 ** 2
t81 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + 0.5e1 / 0.648e3 * t38 * s0 / t41 / t39 + 0.5e1 / 0.54e2 * t38 * l0 / t41 / r0 + t55 * t56 / t40 / t39 / r0 / 0.5832e4 - t55 * s0 / t40 / t63 * l0 / 0.5184e4 + t55 * t70 / t40 / t63 / r0 / 0.17496e5))
t83 = lax_cond(t11, t16, -t18)
t84 = lax_cond(t15, t12, t83)
t85 = 0.1e1 + t84
t87 = t85 ** (0.1e1 / 0.3e1)
t88 = t87 ** 2
t90 = lax_cond(t85 <= p_a_zeta_threshold, t25, t88 * t85)
t92 = r1 ** 2
t93 = r1 ** (0.1e1 / 0.3e1)
t94 = t93 ** 2
t105 = l1 ** 2
t112 = t92 ** 2
t119 = s2 ** 2
t130 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t90 * t31 * (0.1e1 + 0.5e1 / 0.648e3 * t38 * s2 / t94 / t92 + 0.5e1 / 0.54e2 * t38 * l1 / t94 / r1 + t55 * t105 / t93 / t92 / r1 / 0.5832e4 - t55 * s2 / t93 / t112 * l1 / 0.5184e4 + t55 * t119 / t93 / t112 / r1 / 0.17496e5))
res = t81 + t130
