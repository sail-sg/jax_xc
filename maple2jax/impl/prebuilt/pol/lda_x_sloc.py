t1 = params_a_b + 0.1e1
t5 = r0 + r1
t6 = t5 ** params_a_b
t9 = (r0 - r1) / t5
t10 = 0.1e1 + t9
t12 = p_a_zeta_threshold ** t1
t13 = t10 ** t1
t14 = jnp.where(t10 <= p_a_zeta_threshold, t12, t13)
t15 = 0.1e1 - t9
t17 = t15 ** t1
t18 = jnp.where(t15 <= p_a_zeta_threshold, t12, t17)
res = -params_a_a / t1 * t6 * (t14 + t18) / 0.2e1
