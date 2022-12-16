t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t23 = 6 ** (0.1e1 / 0.3e1)
t24 = math.pi ** 2
t25 = t24 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t27 = 0.1e1 / t26
t30 = 2 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t32 = s0 * t31
t33 = r0 ** 2
t34 = t19 ** 2
t36 = 0.1e1 / t34 / t33
t39 = t32 * t36
t62 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + (params_a_muGE + (params_a_muPBE - params_a_muGE) * params_a_alpha * t23 * t27 * t32 * t36 / (0.1e1 + params_a_alpha * t23 * t27 * t39 / 0.24e2) / 0.24e2) * t23 * t27 * t39 / 0.24e2))))
res = 0.2e1 * t62
