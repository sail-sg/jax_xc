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
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t35 = t21 / t24 * s0 * t28 / t31 / t30
t43 = t21 ** 2
t45 = 0.3e1 / 0.10e2 * t43 * t24
t49 = tau0 * t28 / t31 / r0
t50 = t45 - t49
t52 = t45 + t49
t53 = 0.1e1 / t52
t56 = t50 ** 2
t58 = t52 ** 2
t59 = 0.1e1 / t58
t62 = t56 * t50
t64 = t58 * t52
t65 = 0.1e1 / t64
t68 = t56 ** 2
t70 = t58 ** 2
t71 = 0.1e1 / t70
t74 = t68 * t50
t77 = 0.1e1 / t70 / t52
t80 = t68 * t56
t83 = 0.1e1 / t70 / t58
t86 = t68 * t62
t89 = 0.1e1 / t70 / t64
t92 = t68 ** 2
t94 = t70 ** 2
t95 = 0.1e1 / t94
t98 = t92 * t50
t101 = 0.1e1 / t94 / t52
t104 = t92 * t56
t107 = 0.1e1 / t94 / t58
t110 = t92 * t62
t113 = 0.1e1 / t94 / t64
t115 = params_a_a[0] + params_a_a[1] * t50 * t53 + params_a_a[2] * t56 * t59 + params_a_a[3] * t62 * t65 + params_a_a[4] * t68 * t71 + params_a_a[5] * t74 * t77 + params_a_a[6] * t80 * t83 + params_a_a[7] * t86 * t89 + params_a_a[8] * t92 * t95 + params_a_a[9] * t98 * t101 + params_a_a[10] * t104 * t107 + params_a_a[11] * t110 * t113
t118 = math.exp(-0.93189002206715572255e-2 * t35)
t155 = params_a_b[0] + params_a_b[1] * t50 * t53 + params_a_b[2] * t56 * t59 + params_a_b[3] * t62 * t65 + params_a_b[4] * t68 * t71 + params_a_b[5] * t74 * t77 + params_a_b[6] * t80 * t83 + params_a_b[7] * t86 * t89 + params_a_b[8] * t92 * t95 + params_a_b[9] * t98 * t101 + params_a_b[10] * t104 * t107 + params_a_b[11] * t110 * t113
t161 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91462500000000000000e-2 * t35)) * t115 + (0.1552e1 - 0.552e0 * t118) * t155))
res = 0.2e1 * t161
