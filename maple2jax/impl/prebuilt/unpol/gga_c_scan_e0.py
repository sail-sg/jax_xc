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
t58 = math.log(0.2e1)
t59 = 0.1e1 - t58
t60 = math.pi ** 2
t63 = t34 ** 2
t64 = lax_cond(t33, t63, 1)
t65 = t64 ** 2
t66 = t65 * t64
t73 = 0.1e1 / t59
t80 = math.exp(-(-t32 + t57) * t73 * t60 / t66)
t81 = t80 - 0.1e1
t86 = r0 ** 2
t99 = (0.1e1 + 0.27801896084645508333e-2 * (0.1e1 + 0.25000000000000000000e-1 * t10) / (0.1e1 + 0.44450000000000000000e-1 * t10) * t73 * t60 / t81 * s0 / t7 / t86 * t39 / t65 * t18 / t3 * t5) ** (0.1e1 / 0.4e1)
t105 = math.log(0.1e1 + 0.10000000000000000000e1 * (0.1e1 - 0.1e1 / t99) * t81)
res = -t32 + t57 + t59 / t60 * t66 * t105
