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
t33 = 0.1e1 / t31 / t30
t34 = s0 * t28 * t33
t43 = t21 ** 2
t44 = t43 * t24
t45 = 0.3e1 / 0.10e2 * t44
t49 = tau0 * t28 / t31 / r0
t50 = t45 - t49
t52 = t45 + t49
t56 = t50 ** 2
t58 = t52 ** 2
t62 = t56 * t50
t64 = t58 * t52
t68 = t56 ** 2
t70 = t58 ** 2
t92 = t68 ** 2
t94 = t70 ** 2
t115 = params_a_a[0] + params_a_a[1] * t50 / t52 + params_a_a[2] * t56 / t58 + params_a_a[3] * t62 / t64 + params_a_a[4] * t68 / t70 + params_a_a[5] * t68 * t50 / t70 / t52 + params_a_a[6] * t68 * t56 / t70 / t58 + params_a_a[7] * t68 * t62 / t70 / t64 + params_a_a[8] * t92 / t94 + params_a_a[9] * t92 * t50 / t94 / t52 + params_a_a[10] * t92 * t56 / t94 / t58 + params_a_a[11] * t92 * t62 / t94 / t64
t121 = 0.1e1 + 0.186726e-2 * t34 + 0.373452e-2 * t49 - 0.11203560000000000000e-2 * t44
t126 = t28 * t33
t131 = 0.2e1 * t49 - 0.3e1 / 0.5e1 * t44
t134 = t121 ** 2
t138 = s0 ** 2
t140 = t30 ** 2
t152 = t131 ** 2
t162 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t21 / t24 * t34)) * t115 + params_a_d[0] / t121 + (params_a_d[1] * s0 * t126 + params_a_d[2] * t131) / t134 + (0.2e1 * params_a_d[3] * t138 * t27 / t19 / t140 / r0 + params_a_d[4] * s0 * t126 * t131 + params_a_d[5] * t152) / t134 / t121))
res = 0.2e1 * t162
