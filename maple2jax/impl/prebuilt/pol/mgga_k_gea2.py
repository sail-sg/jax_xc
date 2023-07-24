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
t56 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + 0.5e1 / 0.648e3 * t38 * s0 / t41 / t39 + 0.5e1 / 0.54e2 * t38 * l0 / t41 / r0))
t58 = lax_cond(t11, t16, -t18)
t59 = lax_cond(t15, t12, t58)
t60 = 0.1e1 + t59
t62 = t60 ** (0.1e1 / 0.3e1)
t63 = t62 ** 2
t65 = lax_cond(t60 <= p_a_zeta_threshold, t25, t63 * t60)
t67 = r1 ** 2
t68 = r1 ** (0.1e1 / 0.3e1)
t69 = t68 ** 2
t84 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t65 * t31 * (0.1e1 + 0.5e1 / 0.648e3 * t38 * s2 / t69 / t67 + 0.5e1 / 0.54e2 * t38 * l1 / t69 / r1))
res = t56 + t84
