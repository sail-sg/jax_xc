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
t21 = t3 ** 2
t24 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t26 = 4 ** (0.1e1 / 0.3e1)
t28 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t35 = 2 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = r0 ** 2
t39 = t20 ** 2
t50 = 0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + params_a_mu * t28 / t32 * s0 * t36 / t39 / t38 / 0.24e2))
t54 = math.sqrt(math.pi * t21 / t24 * t26 / t50)
t58 = (t12 * r0) ** (0.1e1 / 0.3e1)
t62 = p_a_cam_omega / t54 * t35 / t58 / 0.2e1
t64 = 0.135e1 < t62
t65 = lax_cond(t64, t62, 0.135e1)
t66 = t65 ** 2
t69 = t66 ** 2
t72 = t69 * t66
t75 = t69 ** 2
t87 = t75 ** 2
t91 = lax_cond(t64, 0.135e1, t62)
t92 = math.sqrt(math.pi)
t95 = math.erf(0.1e1 / t91 / 0.2e1)
t97 = t91 ** 2
t100 = math.exp(-0.1e1 / t97 / 0.4e1)
t111 = lax_cond(0.135e1 <= t62, 0.1e1 / t66 / 0.36e2 - 0.1e1 / t69 / 0.960e3 + 0.1e1 / t72 / 0.26880e5 - 0.1e1 / t75 / 0.829440e6 + 0.1e1 / t75 / t66 / 0.28385280e8 - 0.1e1 / t75 / t69 / 0.1073479680e10 + 0.1e1 / t75 / t72 / 0.44590694400e11 - 0.1e1 / t87 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t91 * (t92 * t95 + 0.2e1 * t91 * (t100 - 0.3e1 / 0.2e1 - 0.2e1 * t97 * (t100 - 0.1e1))))
t116 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t111 * t50)
res = 0.2e1 * t116
