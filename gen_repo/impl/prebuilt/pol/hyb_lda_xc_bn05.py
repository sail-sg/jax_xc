t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t4 = t1 * t3
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = t4 * t6
t8 = 2 ** (0.1e1 / 0.3e1)
t9 = t8 ** 2
t10 = r0 - r1
t11 = r0 + r1
t13 = t10 / t11
t14 = 0.1e1 + t13
t15 = t14 <= p_a_zeta_threshold
t16 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t17 = t16 * p_a_zeta_threshold
t18 = t14 ** (0.1e1 / 0.3e1)
t20 = lax_cond(t15, t17, t18 * t14)
t22 = t11 ** (0.1e1 / 0.3e1)
t23 = 9 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t3 ** 2
t27 = t24 * t25 * p_a_cam_omega
t28 = 0.1e1 / t22
t29 = t1 * t28
t30 = lax_cond(t15, t16, t18)
t34 = t27 * t29 / t30 / 0.18e2
t36 = 0.192e1 < t34
t37 = lax_cond(t36, t34, 0.192e1)
t38 = t37 ** 2
t41 = t38 ** 2
t44 = t41 * t38
t47 = t41 ** 2
t50 = t47 * t38
t53 = t47 * t41
t56 = t47 * t44
t59 = t47 ** 2
t83 = t59 ** 2
t92 = 0.1e1 / t38 / 0.9e1 - 0.1e1 / t41 / 0.30e2 + 0.1e1 / t44 / 0.70e2 - 0.1e1 / t47 / 0.135e3 + 0.1e1 / t50 / 0.231e3 - 0.1e1 / t53 / 0.364e3 + 0.1e1 / t56 / 0.540e3 - 0.1e1 / t59 / 0.765e3 + 0.1e1 / t59 / t38 / 0.1045e4 - 0.1e1 / t59 / t41 / 0.1386e4 + 0.1e1 / t59 / t44 / 0.1794e4 - 0.1e1 / t59 / t47 / 0.2275e4 + 0.1e1 / t59 / t50 / 0.2835e4 - 0.1e1 / t59 / t53 / 0.3480e4 + 0.1e1 / t59 / t56 / 0.4216e4 - 0.1e1 / t83 / 0.5049e4 + 0.1e1 / t83 / t38 / 0.5985e4 - 0.1e1 / t83 / t41 / 0.7030e4
t93 = lax_cond(t36, 0.192e1, t34)
t94 = math.atan2(0.1e1, t93)
t95 = t93 ** 2
t99 = math.log(0.1e1 + 0.1e1 / t95)
t108 = lax_cond(0.192e1 <= t34, t92, 0.1e1 - 0.8e1 / 0.3e1 * t93 * (t94 + t93 * (0.1e1 - (t95 + 0.3e1) * t99) / 0.4e1))
t113 = 0.1e1 - t13
t114 = t113 <= p_a_zeta_threshold
t115 = t113 ** (0.1e1 / 0.3e1)
t117 = lax_cond(t114, t17, t115 * t113)
t119 = lax_cond(t114, t16, t115)
t123 = t27 * t29 / t119 / 0.18e2
t125 = 0.192e1 < t123
t126 = lax_cond(t125, t123, 0.192e1)
t127 = t126 ** 2
t130 = t127 ** 2
t133 = t130 * t127
t136 = t130 ** 2
t139 = t136 * t127
t142 = t136 * t130
t145 = t136 * t133
t148 = t136 ** 2
t172 = t148 ** 2
t181 = 0.1e1 / t127 / 0.9e1 - 0.1e1 / t130 / 0.30e2 + 0.1e1 / t133 / 0.70e2 - 0.1e1 / t136 / 0.135e3 + 0.1e1 / t139 / 0.231e3 - 0.1e1 / t142 / 0.364e3 + 0.1e1 / t145 / 0.540e3 - 0.1e1 / t148 / 0.765e3 + 0.1e1 / t148 / t127 / 0.1045e4 - 0.1e1 / t148 / t130 / 0.1386e4 + 0.1e1 / t148 / t133 / 0.1794e4 - 0.1e1 / t148 / t136 / 0.2275e4 + 0.1e1 / t148 / t139 / 0.2835e4 - 0.1e1 / t148 / t142 / 0.3480e4 + 0.1e1 / t148 / t145 / 0.4216e4 - 0.1e1 / t172 / 0.5049e4 + 0.1e1 / t172 / t127 / 0.5985e4 - 0.1e1 / t172 / t130 / 0.7030e4
t182 = lax_cond(t125, 0.192e1, t123)
t183 = math.atan2(0.1e1, t182)
t184 = t182 ** 2
t188 = math.log(0.1e1 + 0.1e1 / t184)
t197 = lax_cond(0.192e1 <= t123, t181, 0.1e1 - 0.8e1 / 0.3e1 * t182 * (t183 + t182 * (0.1e1 - (t184 + 0.3e1) * t188) / 0.4e1))
t203 = t4 * t6 * t28
t206 = math.sqrt(t203)
t209 = t203 ** 0.15e1
t211 = t1 ** 2
t213 = t22 ** 2
t216 = t211 * t25 * t5 / t213
t222 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t206 + 0.89690000000000000000e0 * t203 + 0.20477500000000000000e0 * t209 + 0.12323500000000000000e0 * t216))
t224 = 0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t203) * t222
t225 = t10 ** 2
t226 = t225 ** 2
t227 = t11 ** 2
t228 = t227 ** 2
t235 = (t20 + t117 - 0.2e1) / (0.2e1 * t8 - 0.2e1)
t246 = math.log(0.1e1 + 0.32163958997385070134e2 / (0.70594500000000000000e1 * t206 + 0.15494250000000000000e1 * t203 + 0.42077500000000000000e0 * t209 + 0.15629250000000000000e0 * t216))
t259 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t206 + 0.90577500000000000000e0 * t203 + 0.11003250000000000000e0 * t209 + 0.12417750000000000000e0 * t216))
t260 = (0.1e1 + 0.27812500000000000000e-1 * t203) * t259
res = -0.3e1 / 0.32e2 * t7 * t9 * t20 * t22 * t108 - 0.3e1 / 0.32e2 * t7 * t9 * t117 * t22 * t197 + 0.34602e1 * (-t224 + t226 / t228 * t235 * (-0.3109070e-1 * (0.1e1 + 0.51370000000000000000e-1 * t203) * t246 + t224 - 0.19751673498613801407e-1 * t260) + 0.19751673498613801407e-1 * t235 * t260) / (0.32e1 - 0.22500000000000000000e0 * t203 + t216 / 0.4e1)
