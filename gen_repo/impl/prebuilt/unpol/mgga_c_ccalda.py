t2 = r0 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t7 = r0 ** 2
t12 = tau0 / t3 / r0 - s0 / t3 / t7 / 0.8e1
t14 = 6 ** (0.1e1 / 0.3e1)
t15 = (0.1e1 + params_a_c) * t12 * t14
t16 = math.pi ** 2
t17 = t16 ** (0.1e1 / 0.3e1)
t18 = t17 ** 2
t19 = 0.1e1 / t18
t20 = 2 ** (0.1e1 / 0.3e1)
t21 = t20 ** 2
t22 = t19 * t21
t29 = 0.1e1 / (0.1e1 + 0.5e1 / 0.9e1 * params_a_c * t12 * t14 * t19 * t21)
t30 = 3 ** (0.1e1 / 0.3e1)
t32 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t34 = 4 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t38 = t30 * t32 * t35 / t2
t41 = math.sqrt(t38)
t44 = t38 ** 0.15e1
t46 = t30 ** 2
t47 = t32 ** 2
t51 = t46 * t47 * t34 / t3
t57 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t41 + 0.89690000000000000000e0 * t38 + 0.20477500000000000000e0 * t44 + 0.12323500000000000000e0 * t51))
t61 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t63 = lax_cond(0.1e1 <= p_a_zeta_threshold, t61 * p_a_zeta_threshold, 1)
t80 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t41 + 0.90577500000000000000e0 * t38 + 0.11003250000000000000e0 * t44 + 0.12417750000000000000e0 * t51))
t84 = -0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t38) * t57 + 0.19751673498613801407e-1 * (0.2e1 * t63 - 0.2e1) / (0.2e1 * t20 - 0.2e1) * (0.1e1 + 0.27812500000000000000e-1 * t38) * t80
res = 0.5e1 / 0.9e1 * t15 * t22 * t29 * t84 + (0.1e1 - 0.5e1 / 0.9e1 * t15 * t22 * t29) * t84
