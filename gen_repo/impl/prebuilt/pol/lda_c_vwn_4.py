t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t4 = t1 * t3
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 + r1
t8 = t7 ** (0.1e1 / 0.3e1)
t10 = t6 / t8
t11 = t4 * t10
t12 = t11 / 0.4e1
t13 = math.sqrt(t11)
t16 = 0.1e1 / (t12 + 0.18637200000000000000e1 * t13 + 0.129352e2)
t20 = math.log(t4 * t10 * t16 / 0.4e1)
t21 = 0.310907e-1 * t20
t25 = math.atan(0.61519908197590802322e1 / (t13 + 0.372744e1))
t26 = 0.38783294878113014393e-1 * t25
t27 = t13 / 0.2e1
t29 = (t27 + 0.10498e0) ** 2
t31 = math.log(t29 * t16)
t32 = 0.96902277115443742139e-3 * t31
t33 = math.pi ** 2
t37 = 0.1e1 / (t12 + 0.53417500000000000000e0 * t13 + 0.114813e2)
t41 = math.log(t4 * t10 * t37 / 0.4e1)
t45 = math.atan(0.66920720466459414830e1 / (t13 + 0.106835e1))
t48 = (t27 + 0.228344e0) ** 2
t50 = math.log(t48 * t37)
t54 = r0 - r1
t56 = t54 / t7
t57 = 0.1e1 + t56
t59 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t60 = t59 * p_a_zeta_threshold
t61 = t57 ** (0.1e1 / 0.3e1)
t63 = lax_cond(t57 <= p_a_zeta_threshold, t60, t61 * t57)
t64 = 0.1e1 - t56
t66 = t64 ** (0.1e1 / 0.3e1)
t68 = lax_cond(t64 <= p_a_zeta_threshold, t60, t66 * t64)
t69 = t63 + t68 - 0.2e1
t71 = 2 ** (0.1e1 / 0.3e1)
t72 = t71 - 0.1e1
t74 = 0.1e1 / t72 / 0.2e1
t75 = t54 ** 2
t76 = t75 ** 2
t77 = t7 ** 2
t78 = t77 ** 2
t79 = 0.1e1 / t78
t89 = 0.1e1 / (t12 + 0.35302100000000000000e1 * t13 + 0.180578e2)
t93 = math.log(t4 * t10 * t89 / 0.4e1)
t98 = math.atan(0.47309269095601128300e1 / (t13 + 0.706042e1))
t101 = (t27 + 0.32500e0) ** 2
t103 = math.log(t101 * t89)
res = t21 + t26 + t32 - 0.3e1 / 0.8e1 / t33 * (t41 + 0.32323836906055067299e0 * t45 + 0.21608710360898267022e-1 * t50) * t69 * t74 * (-t76 * t79 + 0.1e1) * t72 + (0.1554535e-1 * t93 + 0.52491393169780936218e-1 * t98 + 0.22478670955426118383e-2 * t103 - t21 - t26 - t32) * t69 * t74 * t76 * t79
