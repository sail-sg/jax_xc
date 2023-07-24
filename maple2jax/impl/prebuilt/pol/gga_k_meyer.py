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
t48 = t33 ** 2
t50 = t48 / t35
t51 = math.sqrt(s0)
t52 = t40 * r0
t56 = t50 * t51 / t52 / 0.72e2
t59 = abs(0.1e1 - t56)
t62 = math.log((0.1e1 + t56) / t59)
t69 = 0.3e1 * (0.1e1 - t38 * s0 / t41 / t39 / 0.864e3) * t62 * t33 * t35 / t51 * t52
t79 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + 0.20e2 * (0.1e1 / 0.2e1 - t69) / (0.1e1 / 0.2e1 + t69)))
t81 = lax_cond(t11, t16, -t18)
t82 = lax_cond(t15, t12, t81)
t83 = 0.1e1 + t82
t85 = t83 ** (0.1e1 / 0.3e1)
t86 = t85 ** 2
t88 = lax_cond(t83 <= p_a_zeta_threshold, t25, t86 * t83)
t90 = r1 ** 2
t91 = r1 ** (0.1e1 / 0.3e1)
t92 = t91 ** 2
t99 = math.sqrt(s2)
t100 = t91 * r1
t104 = t50 * t99 / t100 / 0.72e2
t107 = abs(0.1e1 - t104)
t110 = math.log((0.1e1 + t104) / t107)
t117 = 0.3e1 * (0.1e1 - t38 * s2 / t92 / t90 / 0.864e3) * t110 * t33 * t35 / t99 * t100
t127 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t88 * t31 * (0.1e1 + 0.20e2 * (0.1e1 / 0.2e1 - t117) / (0.1e1 / 0.2e1 + t117)))
res = t79 + t127
