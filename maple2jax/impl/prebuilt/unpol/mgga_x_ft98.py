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
t19 = r0 ** (0.1e1 / 0.3e1)
t22 = 2 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t24 = r0 ** 2
t25 = t19 ** 2
t27 = 0.1e1 / t25 / t24
t28 = t23 * t27
t31 = math.sqrt(params_a_a1 * s0 * t28 + 0.1e1)
t36 = (params_a_b1 * s0 * t28 + 0.1e1) ** (0.1e1 / 0.4e1)
t37 = t36 ** 2
t42 = s0 * t23 * t27
t49 = (t42 - l0 * t23 / t25 / r0) ** 2
t52 = (0.1e1 + t42) ** 2
t57 = params_a_b2 ** 2
t59 = math.sqrt(t57 + 0.1e1)
t60 = t59 - params_a_b2
t61 = s0 ** 2
t63 = t24 ** 2
t67 = t61 * t22 / t19 / t63 / r0
t68 = 0.2e1 * t67
t69 = l0 ** 2
t74 = t69 * t22 / t19 / t24 / r0
t75 = 0.2e1 * t74
t76 = t68 - t75 - params_a_b2
t77 = DBL_EPSILON ** (0.1e1 / 0.4e1)
t78 = 0.1e1 / t77
t88 = lax_cond(0.0e0 < t76, t76, -t76)
t90 = t76 ** 2
t92 = t90 ** 2
t96 = lax_cond(-t78 < t76, t76, -t78)
t97 = t96 ** 2
t99 = math.sqrt(0.1e1 + t97)
t102 = lax_cond(t88 < t77, 0.1e1 - t68 + t75 + params_a_b2 + t90 / 0.2e1 - t92 / 0.8e1, 0.1e1 / (t96 + t99))
t103 = lax_cond(t76 < -t78, -0.4e1 * t67 + 0.4e1 * t74 + 0.2e1 * params_a_b2 - 0.1e1 / t76 / 0.2e1, t102)
t109 = 0.1e1 + (t22 - 0.1e1) * t60 * t103
t110 = t109 ** 2
t117 = t3 ** 2
t119 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t120 = t119 ** 2
t122 = 4 ** (0.1e1 / 0.3e1)
t131 = math.sqrt((0.1e1 + params_a_a * t31 / t37 / t36 * t42 + params_a_b * (0.1e1 + params_a_a2 * t49 / t52) * (t60 * t103 + 0.1e1) / t110 / t109 * t49) / (0.1e1 + 0.81e2 / 0.4e1 * t117 * t120 * t122 * params_a_b * s0 * t28))
t135 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t131)
res = 0.2e1 * t135
