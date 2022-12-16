t1 = params_a_b + 0.1e1
t5 = r0 ** params_a_b
t7 = p_a_zeta_threshold ** t1
t8 = jnp.where(0.1e1 <= p_a_zeta_threshold, t7, 1)
res = -params_a_a / t1 * t5 * t8
