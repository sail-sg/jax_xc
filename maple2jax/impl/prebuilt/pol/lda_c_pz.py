t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 + r1
t8 = t7 ** (0.1e1 / 0.3e1)
t9 = 0.1e1 / t8
t10 = t6 * t9
t11 = t1 * t3 * t10
t12 = t11 / 0.4e1
t13 = 0.1e1 <= t12
t16 = math.sqrt(t11)
t22 = t3 * t6 * t9
t29 = math.log(t12)
t35 = t10 * t29
t43 = jnp.where(t13, params_a_gamma[0] / (0.1e1 + params_a_beta1[0] * t16 / 0.2e1 + params_a_beta2[0] * t1 * t22 / 0.4e1), params_a_a[0] * t29 + params_a_b[0] + params_a_c[0] * t1 * t3 * t35 / 0.4e1 + params_a_d[0] * t1 * t22 / 0.4e1)
t68 = jnp.where(t13, params_a_gamma[1] / (0.1e1 + params_a_beta1[1] * t16 / 0.2e1 + params_a_beta2[1] * t1 * t22 / 0.4e1), params_a_a[1] * t29 + params_a_b[1] + params_a_c[1] * t1 * t3 * t35 / 0.4e1 + params_a_d[1] * t1 * t22 / 0.4e1)
t72 = (r0 - r1) / t7
t73 = 0.1e1 + t72
t75 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t76 = t75 * p_a_zeta_threshold
t77 = t73 ** (0.1e1 / 0.3e1)
t79 = jnp.where(t73 <= p_a_zeta_threshold, t76, t77 * t73)
t80 = 0.1e1 - t72
t82 = t80 ** (0.1e1 / 0.3e1)
t84 = jnp.where(t80 <= p_a_zeta_threshold, t76, t82 * t80)
t87 = 2 ** (0.1e1 / 0.3e1)
res = t43 + (t68 - t43) * (t79 + t84 - 0.2e1) / (0.2e1 * t87 - 0.2e1)
