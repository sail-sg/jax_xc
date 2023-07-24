t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = lax_cond(t14, t15, t17)
t19 = lax_cond(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = params_a_aa * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t44 = t29 ** 2
t45 = params_a_bb * t44
t47 = 0.1e1 / t32 / t31
t48 = s0 ** 2
t50 = t36 ** 2
t57 = t31 ** 2
t59 = params_a_cc / t57
t61 = t50 ** 2
t67 = (0.1e1 + t30 * t34 * s0 / t38 / t36 / 0.24e2 + t45 * t47 * t48 / t37 / t50 / r0 / 0.576e3 + t59 * t48 * s0 / t61 / 0.2304e4) ** (0.1e1 / 0.15e2)
t71 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t67)
t73 = lax_cond(t10, t15, -t17)
t74 = lax_cond(t14, t11, t73)
t75 = 0.1e1 + t74
t77 = t75 ** (0.1e1 / 0.3e1)
t79 = lax_cond(t75 <= p_a_zeta_threshold, t23, t77 * t75)
t82 = r1 ** 2
t83 = r1 ** (0.1e1 / 0.3e1)
t84 = t83 ** 2
t90 = s2 ** 2
t92 = t82 ** 2
t100 = t92 ** 2
t106 = (0.1e1 + t30 * t34 * s2 / t84 / t82 / 0.24e2 + t45 * t47 * t90 / t83 / t92 / r1 / 0.576e3 + t59 * t90 * s2 / t100 / 0.2304e4) ** (0.1e1 / 0.15e2)
t110 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t79 * t27 * t106)
res = t71 + t110
