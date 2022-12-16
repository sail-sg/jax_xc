t1 = math.sqrt(0.2e1)
t2 = math.sqrt(math.pi)
t6 = math.sqrt(p_a_zeta_threshold)
t8 = jnp.where(0.1e1 <= p_a_zeta_threshold, t6 * p_a_zeta_threshold, 1)
t9 = math.sqrt(r0)
res = -0.4e1 / 0.3e1 * t1 / t2 * t8 * t9
