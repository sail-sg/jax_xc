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
t22 = 6 ** (0.1e1 / 0.3e1)
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t31 = r0 ** 2
t32 = t20 ** 2
t44 = t22 ** 2
t46 = 0.3e1 / 0.10e2 * t44 * t25
t50 = tau0 * t29 / t32 / r0
t51 = t46 - t50
t53 = t46 + t50
t57 = t51 ** 2
t59 = t53 ** 2
t63 = t57 * t51
t65 = t59 * t53
t69 = t57 ** 2
t71 = t59 ** 2
t93 = t69 ** 2
t95 = t71 ** 2
t116 = params_a_a[0] + params_a_a[1] * t51 / t53 + params_a_a[2] * t57 / t59 + params_a_a[3] * t63 / t65 + params_a_a[4] * t69 / t71 + params_a_a[5] * t69 * t51 / t71 / t53 + params_a_a[6] * t69 * t57 / t71 / t59 + params_a_a[7] * t69 * t63 / t71 / t65 + params_a_a[8] * t93 / t95 + params_a_a[9] * t93 * t51 / t95 / t53 + params_a_a[10] * t93 * t57 / t95 / t59 + params_a_a[11] * t93 * t63 / t95 / t65
t121 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * params_a_csi_HF * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t22 / t25 * s0 * t29 / t32 / t31)) * t116)
res = 0.2e1 * t121
