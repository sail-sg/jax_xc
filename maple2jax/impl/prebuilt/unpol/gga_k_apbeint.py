t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = jnp.where(t8, -t9, 0)
t12 = jnp.where(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = jnp.where(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t27 = 6 ** (0.1e1 / 0.3e1)
t28 = math.pi ** 2
t29 = t28 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t31 = 0.1e1 / t30
t34 = 2 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t36 = s0 * t35
t37 = r0 ** 2
t39 = 0.1e1 / t23 / t37
t42 = t36 * t39
t65 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + (params_a_muGE + (params_a_muPBE - params_a_muGE) * params_a_alpha * t27 * t31 * t36 * t39 / (0.1e1 + params_a_alpha * t27 * t31 * t42 / 0.24e2) / 0.24e2) * t27 * t31 * t42 / 0.24e2))))
res = 0.2e1 * t65
