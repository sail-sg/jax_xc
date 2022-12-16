t3 = 0.1e1 <= p_a_zeta_threshold
t4 = p_a_zeta_threshold - 0.1e1
t6 = jnp.where(t3, -t4, 0)
t7 = jnp.where(t3, t4, t6)
t8 = 0.1e1 + t7
t10 = math.log(t8 * r0)
t12 = t10 ** 2
t17 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, t8 * (params_a_B * t10 + params_a_C * t12 + params_a_A) / 0.2e1)
res = 0.2e1 * t17
