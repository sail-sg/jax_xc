t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t23 = 6 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = math.pi ** 2
t26 = t25 ** (0.1e1 / 0.3e1)
t30 = s0 ** 2
t32 = 2 ** (0.1e1 / 0.3e1)
t33 = r0 ** 2
t34 = t33 ** 2
t39 = params_a_dp2 ** 2
t40 = t39 ** 2
t45 = math.exp(-t24 / t26 / t25 * t30 * t32 / t20 / t34 / r0 / t40 / 0.288e3)
t50 = t26 ** 2
t53 = t32 ** 2
t55 = t20 ** 2
t57 = 0.1e1 / t55 / t33
t58 = s0 * t53 * t57
t65 = params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + (-0.162742215233874e0 * (0.20e2 / 0.27e2 + 0.5e1 / 0.3e1 * params_a_eta) * t45 + 0.10e2 / 0.81e2) * t23 / t50 * t58 / 0.24e2))
t80 = (tau0 * t53 / t55 / r0 - t58 / 0.8e1) / (0.3e1 / 0.10e2 * t24 * t50 + params_a_eta * s0 * t53 * t57 / 0.8e1)
t83 = lax_cond(0.0e0 < t80, 0, t80)
t88 = math.exp(-params_a_c1 * t83 / (0.1e1 - t83))
t90 = 0.25e1 < t80
t91 = lax_cond(t90, 0.25e1, t80)
t93 = t91 ** 2
t95 = t93 * t91
t97 = t93 ** 2
t106 = lax_cond(t90, t80, 0.25e1)
t110 = math.exp(params_a_c2 / (0.1e1 - t106))
t112 = lax_cond(t80 <= 0.25e1, 0.1e1 - 0.667e0 * t91 - 0.4445555e0 * t93 - 0.663086601049e0 * t95 + 0.1451297044490e1 * t97 - 0.887998041597e0 * t97 * t91 + 0.234528941479e0 * t97 * t93 - 0.23185843322e-1 * t97 * t95, -params_a_d * t110)
t113 = lax_cond(t80 <= 0.0e0, t88, t112)
t118 = math.sqrt(0.3e1)
t121 = math.sqrt(s0)
t127 = math.sqrt(t24 / t26 * t121 * t32 / t20 / r0)
t131 = math.exp(-0.98958000000000000000e1 * t118 / t127)
t136 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + t65 + t113 * (0.174e0 - t65)) * (0.1e1 - t131))
res = 0.2e1 * t136
