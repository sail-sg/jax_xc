t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t8 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t10 = jnp.where(0.1e1 <= p_a_zeta_threshold, t8 * p_a_zeta_threshold, 1)
t11 = r0 ** (0.1e1 / 0.3e1)
t15 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t10 * t11)
res = 0.2e1 * params_a_alpha * t15
