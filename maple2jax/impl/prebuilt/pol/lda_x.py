t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t8 = r0 * t7
t11 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t12 = t11 * p_a_zeta_threshold
t13 = 2 ** (0.1e1 / 0.3e1)
t15 = t8 ** (0.1e1 / 0.3e1)
t19 = jnp.where(0.2e1 * t8 <= p_a_zeta_threshold, t12, 0.2e1 * t13 * r0 * t7 * t15)
t20 = t6 ** (0.1e1 / 0.3e1)
t24 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t19 * t20)
t27 = r1 * t7
t31 = t27 ** (0.1e1 / 0.3e1)
t35 = jnp.where(0.2e1 * t27 <= p_a_zeta_threshold, t12, 0.2e1 * t13 * r1 * t7 * t31)
t39 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t35 * t20)
res = params_a_alpha * t24 + params_a_alpha * t39
