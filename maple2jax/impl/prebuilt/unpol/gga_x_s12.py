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
t20 = r0 ** (0.1e1 / 0.3e1)
t23 = 2 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = r0 ** 2
t26 = t20 ** 2
t29 = t24 / t26 / t25
t31 = s0 ** 2
t33 = t25 ** 2
t54 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * params_a_bx * (params_a_A + params_a_B * (0.1e1 - 0.1e1 / (0.1e1 + params_a_C * s0 * t29 + 0.2e1 * params_a_D * t31 * t23 / t20 / t33 / r0)) * (0.1e1 - 0.1e1 / (params_a_E * s0 * t29 + 0.1e1))))
res = 0.2e1 * t54
