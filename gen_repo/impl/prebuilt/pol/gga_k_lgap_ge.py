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
t34 = 6 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t36 = params_a_mu[0] * t35
t37 = math.pi ** 2
t38 = t37 ** (0.1e1 / 0.3e1)
t39 = 0.1e1 / t38
t40 = math.sqrt(s0)
t42 = r0 ** (0.1e1 / 0.3e1)
t49 = params_a_mu[1] * t34
t50 = t38 ** 2
t51 = 0.1e1 / t50
t53 = r0 ** 2
t54 = t42 ** 2
t62 = params_a_mu[2] / t37
t64 = t53 ** 2
t73 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + t36 * t39 * t40 / t42 / r0 / 0.12e2 + t49 * t51 * s0 / t54 / t53 / 0.24e2 + t62 * t40 * s0 / t64 / 0.48e2))
t75 = lax_cond(t11, t16, -t18)
t76 = lax_cond(t15, t12, t75)
t77 = 0.1e1 + t76
t79 = t77 ** (0.1e1 / 0.3e1)
t80 = t79 ** 2
t82 = lax_cond(t77 <= p_a_zeta_threshold, t25, t80 * t77)
t84 = math.sqrt(s2)
t86 = r1 ** (0.1e1 / 0.3e1)
t93 = r1 ** 2
t94 = t86 ** 2
t101 = t93 ** 2
t110 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t82 * t31 * (0.1e1 + t36 * t39 * t84 / t86 / r1 / 0.12e2 + t49 * t51 * s2 / t94 / t93 / 0.24e2 + t62 * t84 * s2 / t101 / 0.48e2))
res = t73 + t110
