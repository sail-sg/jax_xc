t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 ** (0.1e1 / 0.3e1)
t10 = t1 * t3 * t6 / t7
t13 = math.sqrt(t10)
t16 = t10 ** 0.15e1
t18 = t1 ** 2
t19 = t3 ** 2
t21 = t7 ** 2
t24 = t18 * t19 * t5 / t21
t30 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t13 + 0.89690000000000000000e0 * t10 + 0.20477500000000000000e0 * t16 + 0.12323500000000000000e0 * t24))
t32 = 0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t10) * t30
t33 = 0.1e1 <= p_a_zeta_threshold
t34 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t36 = lax_cond(t33, t34 * p_a_zeta_threshold, 1)
t39 = 2 ** (0.1e1 / 0.3e1)
t54 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t13 + 0.90577500000000000000e0 * t10 + 0.11003250000000000000e0 * t16 + 0.12417750000000000000e0 * t24))
t57 = 0.19751673498613801407e-1 * (0.2e1 * t36 - 0.2e1) / (0.2e1 * t39 - 0.2e1) * (0.1e1 + 0.27812500000000000000e-1 * t10) * t54
t58 = t34 ** 2
t59 = lax_cond(t33, t58, 1)
t60 = math.sqrt(s0)
t62 = r0 ** 2
t63 = t62 ** 2
t66 = t59 ** 2
t67 = t66 * t59
t68 = 0.1e1 / t67
t74 = t59 ** (0.50000000000000000000e-1 * t60 * s0 / t63 * t68 / t13 / t10)
t75 = math.log(0.2e1)
t76 = 0.1e1 - t75
t78 = math.pi ** 2
t79 = 0.1e1 / t78
t85 = t39 ** 2
t91 = math.exp(-t24 / 0.4e1)
t96 = 0.786e0 * t79 + 0.17500000000000000000e-1 * t60 / t7 / r0 * t85 / t59 / t13 * (0.1e1 - t91)
t108 = 0.1e1 / t76
t109 = t96 * t108
t114 = math.exp(-(-t32 + t57) * t108 * t78 * t68)
t117 = t78 / (t114 - 0.1e1)
t118 = s0 ** 2
t124 = t66 ** 2
t133 = s0 / t7 / t62 * t39 / t66 * t18 / t3 * t5 / 0.96e2 + t109 * t117 * t118 / t21 / t63 * t85 / t124 * t1 / t19 * t6 / 0.3072e4
t143 = math.log(0.1e1 + t96 * t133 * t108 * t78 / (t109 * t117 * t133 + 0.1e1))
res = t74 * t76 * t79 * t67 * t143 - t32 + t57
