t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 + r1
t8 = t7 ** (0.1e1 / 0.3e1)
t11 = t1 * t3 * t6 / t8
t14 = math.sqrt(t11)
t17 = t11 ** 0.15e1
t19 = t1 ** 2
t20 = t3 ** 2
t22 = t8 ** 2
t25 = t19 * t20 * t5 / t22
t31 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t14 + 0.89690000000000000000e0 * t11 + 0.20477500000000000000e0 * t17 + 0.12323500000000000000e0 * t25))
t33 = 0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t11) * t31
t34 = r0 - r1
t35 = t34 ** 2
t36 = t35 ** 2
t37 = t7 ** 2
t38 = t37 ** 2
t42 = t34 / t7
t43 = 0.1e1 + t42
t44 = t43 <= p_a_zeta_threshold
t45 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t46 = t45 * p_a_zeta_threshold
t47 = t43 ** (0.1e1 / 0.3e1)
t49 = lax_cond(t44, t46, t47 * t43)
t50 = 0.1e1 - t42
t51 = t50 <= p_a_zeta_threshold
t52 = t50 ** (0.1e1 / 0.3e1)
t54 = lax_cond(t51, t46, t52 * t50)
t56 = 2 ** (0.1e1 / 0.3e1)
t60 = (t49 + t54 - 0.2e1) / (0.2e1 * t56 - 0.2e1)
t71 = math.log(0.1e1 + 0.32163958997385070134e2 / (0.70594500000000000000e1 * t14 + 0.15494250000000000000e1 * t11 + 0.42077500000000000000e0 * t17 + 0.15629250000000000000e0 * t25))
t84 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t14 + 0.90577500000000000000e0 * t11 + 0.11003250000000000000e0 * t17 + 0.12417750000000000000e0 * t25))
t85 = (0.1e1 + 0.27812500000000000000e-1 * t11) * t84
t92 = -t33 + t36 / t38 * t60 * (-0.3109070e-1 * (0.1e1 + 0.51370000000000000000e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85) + 0.19751673498613801407e-1 * t60 * t85
t95 = t45 ** 2
t96 = t47 ** 2
t97 = lax_cond(t44, t95, t96)
t98 = t52 ** 2
t99 = lax_cond(t51, t95, t98)
t115 = 0.69506584583333333332e-3 * t56 * (t97 / 0.2e1 + t99 / 0.2e1) * (s0 + 0.2e1 * s1 + s2) / t8 / t37 * t19 / t3 * t5 / t92
t118 = 0.1e1 - 0.1e1 / (0.1e1 - t115)
t121 = t118 ** 2
t127 = t121 ** 2
t134 = math.exp(t115)
t135 = 0.1e1 - t134
t138 = t135 ** 2
t144 = t138 ** 2
t149 = params_a_sogga11_a[3] * t121 * t118 + params_a_sogga11_a[5] * t127 * t118 + params_a_sogga11_b[3] * t138 * t135 + params_a_sogga11_b[5] * t144 * t135 + params_a_sogga11_a[1] * t118 + params_a_sogga11_a[2] * t121 + params_a_sogga11_a[4] * t127 + params_a_sogga11_b[1] * t135 + params_a_sogga11_b[2] * t138 + params_a_sogga11_b[4] * t144 + params_a_sogga11_a[0] + params_a_sogga11_b[0]
res = t92 * t149
