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
t22 = t19 ** 2
t23 = 0.1e1 / t22
t25 = t12 ** 2
t27 = (t12 * r0) ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t29 = t25 * t28
t38 = params_a_beta0 + params_a_beta1 * s0 * t23 * t29 / (params_a_beta2 + s0 * t23 * t29 / 0.4e1) / 0.4e1
t40 = 2 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t42 = r0 ** 2
t44 = 0.1e1 / t22 / t42
t47 = t3 ** 2
t49 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t52 = 4 ** (0.1e1 / 0.3e1)
t53 = s0 * t41
t54 = t38 ** 2
t57 = math.asinh(t53 * t44)
t58 = t57 ** 2
t63 = math.sqrt(0.9e1 * t53 * t44 * t54 * t58 + 0.1e1)
t73 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.9e1 * t38 * s0 * t41 * t44 * t47 / t49 * t52 / t63))
res = 0.2e1 * t73
