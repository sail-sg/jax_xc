t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = lax_cond(t14, t15, t17)
t19 = lax_cond(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t28 = t6 ** (0.1e1 / 0.3e1)
t29 = t28 * params_a_csi_HF
t30 = 6 ** (0.1e1 / 0.3e1)
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = t30 / t33
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t48 = params_a_a[0]
t49 = params_a_a[1]
t50 = t30 ** 2
t52 = 0.3e1 / 0.10e2 * t50 * t33
t55 = tau0 / t38 / r0
t56 = t52 - t55
t58 = t52 + t55
t61 = params_a_a[2]
t62 = t56 ** 2
t64 = t58 ** 2
t67 = params_a_a[3]
t68 = t62 * t56
t70 = t64 * t58
t73 = params_a_a[4]
t74 = t62 ** 2
t76 = t64 ** 2
t79 = params_a_a[5]
t85 = params_a_a[6]
t91 = params_a_a[7]
t97 = params_a_a[8]
t98 = t74 ** 2
t100 = t76 ** 2
t103 = params_a_a[9]
t109 = params_a_a[10]
t115 = params_a_a[11]
t121 = t48 + t49 * t56 / t58 + t61 * t62 / t64 + t67 * t68 / t70 + t73 * t74 / t76 + t79 * t74 * t56 / t76 / t58 + t85 * t74 * t62 / t76 / t64 + t91 * t74 * t68 / t76 / t70 + t97 * t98 / t100 + t103 * t98 * t56 / t100 / t58 + t109 * t98 * t62 / t100 / t64 + t115 * t98 * t68 / t100 / t70
t126 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t29 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t35 * s0 / t38 / t36)) * t121)
t128 = lax_cond(t10, t15, -t17)
t129 = lax_cond(t14, t11, t128)
t130 = 0.1e1 + t129
t132 = t130 ** (0.1e1 / 0.3e1)
t134 = lax_cond(t130 <= p_a_zeta_threshold, t23, t132 * t130)
t136 = r1 ** 2
t137 = r1 ** (0.1e1 / 0.3e1)
t138 = t137 ** 2
t150 = tau1 / t138 / r1
t151 = t52 - t150
t153 = t52 + t150
t156 = t151 ** 2
t158 = t153 ** 2
t161 = t156 * t151
t163 = t158 * t153
t166 = t156 ** 2
t168 = t158 ** 2
t186 = t166 ** 2
t188 = t168 ** 2
t206 = t48 + t49 * t151 / t153 + t61 * t156 / t158 + t67 * t161 / t163 + t73 * t166 / t168 + t79 * t166 * t151 / t168 / t153 + t85 * t166 * t156 / t168 / t158 + t91 * t166 * t161 / t168 / t163 + t97 * t186 / t188 + t103 * t186 * t151 / t188 / t153 + t109 * t186 * t156 / t188 / t158 + t115 * t186 * t161 / t188 / t163
t211 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t134 * t29 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t35 * s2 / t138 / t136)) * t206)
res = t126 + t211
