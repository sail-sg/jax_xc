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
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = params_a_mu * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t41 = t34 * s0 / t38 / t36
t49 = params_a_kappa + 0.1e1
t50 = params_a_alpha * t29
t53 = math.exp(-t50 * t41 / 0.24e2)
t60 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t30 * t41 / 0.24e2)) - t49 * (0.1e1 - t53)))
t62 = lax_cond(t10, t15, -t17)
t63 = lax_cond(t14, t11, t62)
t64 = 0.1e1 + t63
t66 = t64 ** (0.1e1 / 0.3e1)
t68 = lax_cond(t64 <= p_a_zeta_threshold, t23, t66 * t64)
t71 = r1 ** 2
t72 = r1 ** (0.1e1 / 0.3e1)
t73 = t72 ** 2
t76 = t34 * s2 / t73 / t71
t86 = math.exp(-t50 * t76 / 0.24e2)
t93 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t30 * t76 / 0.24e2)) - t49 * (0.1e1 - t86)))
res = t60 + t93
