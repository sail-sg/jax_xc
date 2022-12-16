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
t22 = 2 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t24 = r0 ** 2
t25 = t19 ** 2
t31 = math.sqrt(s0)
t34 = 0.1e1 / t19 / r0
t47 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (params_a_aa + 0.13888888888888888889e-1 * params_a_bb * s0 * t23 / t25 / t24 + params_a_cc * t31 * t22 * t34 / (0.4e1 * t31 * t22 * t34 + t22)))
res = 0.2e1 * t47
