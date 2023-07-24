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
t47 = math.exp(-0.83254166666666666664e1 * t38 * s0 / t41 / t39)
t49 = t33 ** 2
t52 = t49 / t35 / t34
t53 = s0 ** 2
t54 = t39 ** 2
t61 = math.exp(-0.75479166666666666666e-2 * t52 * t53 / t40 / t54 / r0)
t67 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.20788e1 - 0.8524e0 * t47 - 0.12264e1 * t61))
t69 = lax_cond(t11, t16, -t18)
t70 = lax_cond(t15, t12, t69)
t71 = 0.1e1 + t70
t73 = t71 ** (0.1e1 / 0.3e1)
t74 = t73 ** 2
t76 = lax_cond(t71 <= p_a_zeta_threshold, t25, t74 * t71)
t78 = r1 ** 2
t79 = r1 ** (0.1e1 / 0.3e1)
t80 = t79 ** 2
t86 = math.exp(-0.83254166666666666664e1 * t38 * s2 / t80 / t78)
t88 = s2 ** 2
t89 = t78 ** 2
t96 = math.exp(-0.75479166666666666666e-2 * t52 * t88 / t79 / t89 / r1)
t102 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t76 * t31 * (0.20788e1 - 0.8524e0 * t86 - 0.12264e1 * t96))
res = t67 + t102
