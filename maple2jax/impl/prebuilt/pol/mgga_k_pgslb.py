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
t37 = 0.1e1 / t36
t38 = t33 * t37
t39 = r0 ** 2
t40 = r0 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t43 = 0.1e1 / t41 / t39
t47 = params_a_pgslb_mu * t33
t52 = math.exp(-t47 * t37 * s0 * t43 / 0.24e2)
t53 = t33 ** 2
t54 = params_a_pgslb_beta * t53
t56 = 0.1e1 / t35 / t34
t57 = l0 ** 2
t69 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.5e1 / 0.72e2 * t38 * s0 * t43 + t52 + t54 * t56 * t57 / t40 / t39 / r0 / 0.576e3))
t71 = lax_cond(t11, t16, -t18)
t72 = lax_cond(t15, t12, t71)
t73 = 0.1e1 + t72
t75 = t73 ** (0.1e1 / 0.3e1)
t76 = t75 ** 2
t78 = lax_cond(t73 <= p_a_zeta_threshold, t25, t76 * t73)
t80 = r1 ** 2
t81 = r1 ** (0.1e1 / 0.3e1)
t82 = t81 ** 2
t84 = 0.1e1 / t82 / t80
t92 = math.exp(-t47 * t37 * s2 * t84 / 0.24e2)
t93 = l1 ** 2
t105 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t78 * t31 * (0.5e1 / 0.72e2 * t38 * s2 * t84 + t92 + t54 * t56 * t93 / t81 / t80 / r1 / 0.576e3))
res = t69 + t105
