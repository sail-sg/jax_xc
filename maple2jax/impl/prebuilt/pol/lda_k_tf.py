t2 = r0 + r1
t4 = (r0 - r1) / t2
t5 = 0.1e1 + t4
t7 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t8 = t7 ** 2
t9 = t8 * p_a_zeta_threshold
t10 = t5 ** (0.1e1 / 0.3e1)
t11 = t10 ** 2
t13 = jnp.where(t5 <= p_a_zeta_threshold, t9, t11 * t5)
t14 = 0.1e1 - t4
t16 = t14 ** (0.1e1 / 0.3e1)
t17 = t16 ** 2
t19 = jnp.where(t14 <= p_a_zeta_threshold, t9, t17 * t14)
t23 = 3 ** (0.1e1 / 0.3e1)
t26 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = 4 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t32 = t2 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
res = params_a_ax * (t13 / 0.2e1 + t19 / 0.2e1) * t23 / t27 * t30 * t33 / 0.3e1
