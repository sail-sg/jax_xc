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
t22 = params_a_mu * t21
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t26 = 0.1e1 / t25
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t31 = r0 ** 2
t32 = t19 ** 2
t33 = t32 * t31
t34 = 0.1e1 / t33
t39 = s0 * t30 * t34
t42 = math.exp(-params_a_alpha * t21 * t26 * t39 / 0.24e2)
t52 = t21 ** 2
t57 = s0 ** 2
t59 = t31 ** 2
t66 = math.exp(-params_a_alpha * t52 / t24 / t23 * t57 * t29 / t19 / t59 / r0 / 0.288e3)
t79 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (t22 * t26 * s0 * t30 * t34 * t42 / (0.1e1 + t22 * t26 * t39 / 0.24e2) / 0.24e2 + 0.2e1 * (0.1e1 - t66) * t52 * t25 / s0 * t29 * t33 + t66))
res = 0.2e1 * t79
