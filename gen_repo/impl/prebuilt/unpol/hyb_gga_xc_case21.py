t4 = 3 ** (0.1e1 / 0.3e1)
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = lax_cond(t8, -t9, 0)
t12 = lax_cond(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 * p_a_zeta_threshold
t17 = t13 ** (0.1e1 / 0.3e1)
t19 = lax_cond(t13 <= p_a_zeta_threshold, t16, t17 * t13)
t20 = r0 ** (0.1e1 / 0.3e1)
t22 = 6 ** (0.1e1 / 0.3e1)
t24 = math.pi ** 2
t25 = t24 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t28 = params_a_gammax * t22 / t26
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t31 = s0 * t30
t32 = r0 ** 2
t33 = t20 ** 2
t35 = 0.1e1 / t33 / t32
t45 = xbspline(t28 * t31 * t35 / (0.1e1 + t28 * t31 * t35 / 0.24e2) / 0.24e2, 0, params)
t49 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t4 / t5 * t19 * t20 * t45)
t52 = t15 ** 2
t53 = lax_cond(t8, t52, 1)
t54 = t4 ** 2
t55 = t53 * t54
t58 = 0.1e1 / t20 / t32
t65 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t67 = 4 ** (0.1e1 / 0.3e1)
t68 = t67 ** 2
t71 = t4 * t65 * t68 / t20
t74 = math.sqrt(t71)
t77 = t71 ** 0.15e1
t79 = t65 ** 2
t83 = t54 * t79 * t67 / t33
t89 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t74 + 0.89690000000000000000e0 * t71 + 0.20477500000000000000e0 * t77 + 0.12323500000000000000e0 * t83))
t92 = lax_cond(t8, t16, 1)
t109 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t74 + 0.90577500000000000000e0 * t71 + 0.11003250000000000000e0 * t77 + 0.12417750000000000000e0 * t83))
t113 = -0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t71) * t89 + 0.19751673498613801407e-1 * (0.2e1 * t92 - 0.2e1) / (0.2e1 * t29 - 0.2e1) * (0.1e1 + 0.27812500000000000000e-1 * t71) * t109
t120 = cbspline(-t55 * t5 * s0 * t58 / (-t55 * t5 * s0 * t58 / 0.48e2 + params_a_gammac * t113) / 0.48e2, 0, params)
res = 0.2e1 * (0.1e1 - params_a_ax) * t49 + t120 * t113
