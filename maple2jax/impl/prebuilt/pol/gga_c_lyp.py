t1 = r0 - r1
t2 = t1 ** 2
t3 = r0 + r1
t4 = t3 ** 2
t7 = 0.1e1 - t2 / t4
t8 = t3 ** (0.1e1 / 0.3e1)
t9 = 0.1e1 / t8
t12 = 0.1e1 / (params_a_d * t9 + 0.1e1)
t15 = math.exp(-params_a_c * t9)
t19 = t8 ** 2
t25 = (params_a_d * t12 + params_a_c) * t9
t32 = 3 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = math.pi ** 2
t35 = t34 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t39 = t1 / t3
t40 = 0.1e1 + t39
t41 = t40 <= p_a_zeta_threshold
t42 = p_a_zeta_threshold ** 2
t43 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t45 = t44 * t42
t46 = t40 ** 2
t47 = t40 ** (0.1e1 / 0.3e1)
t48 = t47 ** 2
t50 = lax_cond(t41, t45, t48 * t46)
t51 = 0.1e1 - t39
t52 = t51 <= p_a_zeta_threshold
t53 = t51 ** 2
t54 = t51 ** (0.1e1 / 0.3e1)
t55 = t54 ** 2
t57 = lax_cond(t52, t45, t55 * t53)
t62 = 2 ** (0.1e1 / 0.3e1)
t63 = t62 * t7
t66 = r0 ** 2
t67 = r0 ** (0.1e1 / 0.3e1)
t68 = t67 ** 2
t70 = 0.1e1 / t68 / t66
t71 = s0 * t70
t72 = t71 * t50
t73 = r1 ** 2
t74 = r1 ** (0.1e1 / 0.3e1)
t75 = t74 ** 2
t77 = 0.1e1 / t75 / t73
t78 = s2 * t77
t79 = t78 * t57
t86 = t44 * t42 * p_a_zeta_threshold
t89 = lax_cond(t41, t86, t48 * t46 * t40)
t93 = lax_cond(t52, t86, t55 * t53 * t51)
t101 = lax_cond(t41, t42, t46)
t106 = lax_cond(t52, t42, t53)
res = params_a_a * (-t7 * t12 + params_a_b * t15 * t12 * (-(s0 + 0.2e1 * s1 + s2) / t19 / t4 * (t7 * (0.47e2 - 0.7e1 * t25) / 0.72e2 - 0.2e1 / 0.3e1) - 0.3e1 / 0.20e2 * t33 * t36 * t7 * (t50 + t57) + t63 * (0.5e1 / 0.2e1 - t25 / 0.18e2) * (t72 + t79) / 0.32e2 + t63 * (t25 - 0.11e2) * (t71 * t89 + t78 * t93) / 0.576e3 - t62 * (0.2e1 / 0.3e1 * t72 + 0.2e1 / 0.3e1 * t79 - t101 * s2 * t77 * t57 / 0.4e1 - t106 * s0 * t70 * t50 / 0.4e1) / 0.8e1))
