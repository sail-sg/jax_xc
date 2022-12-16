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
t21 = t3 ** 2
t24 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t26 = 4 ** (0.1e1 / 0.3e1)
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t12 * r0
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 * t30
t34 = r0 ** 2
t35 = t19 ** 2
t39 = t29 ** 2
t52 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 - params_a_gamma * t21 / t24 * t26 * t29 * t32 * s0 / t35 / t34 / (t39 * t32 / 0.4e1 + params_a_delta) / 0.9e1))
res = 0.2e1 * t52
