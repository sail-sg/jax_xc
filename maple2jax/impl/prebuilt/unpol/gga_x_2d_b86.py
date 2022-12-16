t3 = math.sqrt(math.pi)
t5 = 0.1e1 <= p_a_zeta_threshold
t6 = p_a_zeta_threshold - 0.1e1
t8 = jnp.where(t5, -t6, 0)
t9 = jnp.where(t5, t6, t8)
t10 = 0.1e1 + t9
t12 = math.sqrt(p_a_zeta_threshold)
t14 = math.sqrt(t10)
t16 = jnp.where(t10 <= p_a_zeta_threshold, t12 * p_a_zeta_threshold, t14 * t10)
t18 = math.sqrt(0.2e1)
t20 = math.sqrt(r0)
t21 = r0 ** 2
t24 = s0 / t21 / r0
t34 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t20 * (0.1e1 + 0.4210e-2 * t24) / (0.1e1 + 0.238e-3 * t24))
res = 0.2e1 * t34
