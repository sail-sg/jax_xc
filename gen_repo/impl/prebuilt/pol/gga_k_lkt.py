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
t34 = t33 ** 2
t35 = math.pi ** 2
t36 = t35 ** (0.1e1 / 0.3e1)
t38 = t34 / t36
t39 = math.sqrt(s0)
t40 = r0 ** (0.1e1 / 0.3e1)
t45 = t38 * t39 / t40 / r0 / 0.12e2
t47 = lax_cond(t45 < 0.200e3, t45, 200)
t49 = math.cosh(params_a_a * t47)
t51 = t36 ** 2
t53 = t33 / t51
t54 = r0 ** 2
t55 = t40 ** 2
t65 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 / t49 + 0.5e1 / 0.72e2 * t53 * s0 / t55 / t54))
t67 = lax_cond(t11, t16, -t18)
t68 = lax_cond(t15, t12, t67)
t69 = 0.1e1 + t68
t71 = t69 ** (0.1e1 / 0.3e1)
t72 = t71 ** 2
t74 = lax_cond(t69 <= p_a_zeta_threshold, t25, t72 * t69)
t76 = math.sqrt(s2)
t77 = r1 ** (0.1e1 / 0.3e1)
t82 = t38 * t76 / t77 / r1 / 0.12e2
t84 = lax_cond(t82 < 0.200e3, t82, 200)
t86 = math.cosh(params_a_a * t84)
t88 = r1 ** 2
t89 = t77 ** 2
t99 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t74 * t31 * (0.1e1 / t86 + 0.5e1 / 0.72e2 * t53 * s2 / t89 / t88))
res = t65 + t99
