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
t34 = s0 * t28 / t31 / t30
t36 = 0.5e1 / 0.972e3 * t21 / t24 * t34
t41 = params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t36))
t45 = tau0 * t28 / t31 / r0
t47 = t45 - t34 / 0.8e1
t48 = t47 ** 2
t49 = t21 ** 2
t52 = t45 + 0.3e1 / 0.10e2 * t49 * t24
t53 = t52 ** 2
t57 = 0.1e1 - 0.4e1 * t48 / t53
t58 = t57 ** 2
t65 = t48 ** 2
t68 = t53 ** 2
t87 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t41 + t58 * t57 / (0.1e1 + 0.8e1 * t48 * t47 / t53 / t52 + 0.64e2 * params_a_b * t65 * t48 / t68 / t53) * (params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t36 + params_a_c)) - t41)))
res = 0.2e1 * t87
