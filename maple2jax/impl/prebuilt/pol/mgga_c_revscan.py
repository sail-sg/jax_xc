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
t55 = t49 + t54 - 0.2e1
t56 = 2 ** (0.1e1 / 0.3e1)
t57 = t56 - 0.1e1
t59 = 0.1e1 / t57 / 0.2e1
t60 = t55 * t59
t71 = math.log(0.1e1 + 0.32163958997385070134e2 / (0.70594500000000000000e1 * t14 + 0.15494250000000000000e1 * t11 + 0.42077500000000000000e0 * t17 + 0.15629250000000000000e0 * t25))
t84 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t14 + 0.90577500000000000000e0 * t11 + 0.11003250000000000000e0 * t17 + 0.12417750000000000000e0 * t25))
t85 = (0.1e1 + 0.27812500000000000000e-1 * t11) * t84
t89 = t36 / t38 * t60 * (-0.3109070e-1 * (0.1e1 + 0.51370000000000000000e-1 * t11) * t71 + t33 - 0.19751673498613801407e-1 * t85)
t91 = 0.19751673498613801407e-1 * t60 * t85
t92 = math.log(0.2e1)
t93 = 0.1e1 - t92
t94 = math.pi ** 2
t97 = t45 ** 2
t98 = t47 ** 2
t99 = lax_cond(t44, t97, t98)
t100 = t52 ** 2
t101 = lax_cond(t51, t97, t100)
t103 = t99 / 0.2e1 + t101 / 0.2e1
t104 = t103 ** 2
t105 = t104 * t103
t107 = 0.1e1 + 0.25000000000000000000e-1 * t11
t109 = 0.1e1 + 0.44450000000000000000e-1 * t11
t112 = 0.1e1 / t93
t119 = math.exp(-(-t33 + t89 + t91) * t112 * t94 / t105)
t120 = t119 - 0.1e1
t124 = s0 + 0.2e1 * s1 + s2
t139 = (0.1e1 + 0.55603792169291016666e-2 * t107 / t109 * t112 * t94 / t120 * t124 / t8 / t37 * t56 / t104 * t19 / t3 * t5) ** (0.1e1 / 0.4e1)
t142 = t107 ** 2
t143 = t109 ** 2
t146 = t93 ** 2
t149 = t94 ** 2
t150 = t120 ** 2
t153 = t124 ** 2
t158 = t56 ** 2
t160 = t104 ** 2
t170 = (0.1e1 + 0.11594181388521408694e-3 * t142 / t143 / t146 * t149 / t150 * t153 / t22 / t38 * t158 / t160 * t1 / t20 * t6) ** (0.1e1 / 0.8e1)
t177 = math.log(0.1e1 + 0.10000000000000000000e1 * (0.1e1 - 0.1e1 / t139 / 0.2e1 - 0.1e1 / t170 / 0.2e1) * t120)
t179 = t93 / t94 * t105 * t177
t180 = r0 ** (0.1e1 / 0.3e1)
t181 = t180 ** 2
t185 = t43 / 0.2e1
t186 = t185 ** (0.1e1 / 0.3e1)
t187 = t186 ** 2
t188 = t187 * t185
t190 = r1 ** (0.1e1 / 0.3e1)
t191 = t190 ** 2
t195 = t50 / 0.2e1
t196 = t195 ** (0.1e1 / 0.3e1)
t197 = t196 ** 2
t198 = t197 * t195
t201 = 0.1e1 / t22 / t37
t205 = 6 ** (0.1e1 / 0.3e1)
t207 = t94 ** (0.1e1 / 0.3e1)
t208 = t207 ** 2
t209 = 0.1e1 / t208
t214 = 0.5e1 / 0.9e1 * (tau0 / t181 / r0 * t188 + tau1 / t191 / r1 * t198 - t124 * t201 / 0.8e1) * t205 * t209 / (t188 + t198)
t216 = math.log(DBL_EPSILON)
t219 = t216 / (-t216 + 0.1131e1)
t222 = lax_cond(t214 < -t219, t214, -t219)
t227 = math.exp(-0.1131e1 * t222 / (0.1e1 - t222))
t228 = lax_cond(-t219 < t214, 0, t227)
t230 = math.log(0.72992700729927007299e0 * DBL_EPSILON)
t233 = (-t230 + 0.17e1) / t230
t234 = t214 < -t233
t235 = lax_cond(t234, -t233, t214)
t239 = math.exp(0.17e1 / (0.1e1 - t235))
t241 = lax_cond(t234, 0, -0.137e1 * t239)
t242 = lax_cond(t214 <= 0.1e1, t228, t241)
t246 = 0.1e1 / (0.1e1 - 0.33115000000000000000e-1 * t14 + 0.41680000000000000000e-1 * t11)
t249 = math.exp(0.10000000000000000000e1 * t246)
t257 = (0.1e1 + 0.21337642104376358333e-1 * t205 * t209 * t158 * t124 * t201) ** (0.1e1 / 0.4e1)
t260 = t205 ** 2
t272 = (0.1e1 + 0.45529497057445474566e-2 * t260 / t207 / t94 * t56 * t153 / t8 / t38 / t7) ** (0.1e1 / 0.8e1)
t277 = math.log(0.1e1 + (t249 - 0.1e1) * (0.1e1 - 0.1e1 / t257 / 0.2e1 - t272 / 0.2e1))
t285 = t36 ** 2
t287 = t38 ** 2
res = -t33 + t89 + t91 + t179 + t242 * ((-0.30197e-1 * t246 + 0.30197e-1 * t277) * (0.1e1 - 0.2363e1 * t57 * t55 * t59) * (0.1e1 - t285 * t36 / t287 / t38) + t33 - t89 - t91 - t179)
