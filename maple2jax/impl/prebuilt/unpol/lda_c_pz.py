t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 ** (0.1e1 / 0.3e1)
t8 = 0.1e1 / t7
t9 = t6 * t8
t10 = t1 * t3 * t9
t11 = t10 / 0.4e1
t12 = 0.1e1 <= t11
t15 = math.sqrt(t10)
t21 = t3 * t6 * t8
t28 = math.log(t11)
t34 = t9 * t28
t42 = jnp.where(t12, params_a_gamma[0] / (0.1e1 + params_a_beta1[0] * t15 / 0.2e1 + params_a_beta2[0] * t1 * t21 / 0.4e1), params_a_a[0] * t28 + params_a_b[0] + params_a_c[0] * t1 * t3 * t34 / 0.4e1 + params_a_d[0] * t1 * t21 / 0.4e1)
t67 = jnp.where(t12, params_a_gamma[1] / (0.1e1 + params_a_beta1[1] * t15 / 0.2e1 + params_a_beta2[1] * t1 * t21 / 0.4e1), params_a_a[1] * t28 + params_a_b[1] + params_a_c[1] * t1 * t3 * t34 / 0.4e1 + params_a_d[1] * t1 * t21 / 0.4e1)
t70 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t72 = jnp.where(0.1e1 <= p_a_zeta_threshold, t70 * p_a_zeta_threshold, 1)
t76 = 2 ** (0.1e1 / 0.3e1)
res = t42 + (t67 - t42) * (0.2e1 * t72 - 0.2e1) / (0.2e1 * t76 - 0.2e1)
