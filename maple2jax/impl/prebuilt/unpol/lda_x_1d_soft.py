t3 = 0.1e1 <= p_a_zeta_threshold
t4 = jnp.logical_or(r0 / 0.2e1 <= p_a_dens_threshold, t3)
t5 = p_a_zeta_threshold - 0.1e1
t7 = jnp.where(t3, -t5, 0)
t8 = jnp.where(t3, t5, t7)
t9 = 0.1e1 + t8
t12 = t9 * math.pi * params_a_beta * r0
t13 = int1(t12)
t15 = int2(t12)
t16 = 0.1e1 / math.pi
t18 = 0.1e1 / params_a_beta
t26 = jnp.where(t4, 0, -0.25000000000000000000e0 * (t9 * t13 - t15 * t16 * t18 / r0) * t16 * t18)
res = 0.2e1 * t26
