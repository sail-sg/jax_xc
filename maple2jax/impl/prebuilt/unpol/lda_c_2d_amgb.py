t1 = math.sqrt(math.pi)
t2 = 0.1e1 / t1
t3 = math.sqrt(r0)
t5 = t2 / t3
t9 = 0.1e1 / math.pi / r0
t15 = 0.1e1 / t1 / math.pi / t3 / r0
t19 = t5 ** 0.15e1
t26 = math.log(0.1e1 + 0.1e1 / (0.10022e1 * t5 - 0.2069e-1 * t19 + 0.33997e0 * t9 + 0.1747e-1 * t15))
t29 = math.exp(-0.13386e1 * t5)
t31 = math.sqrt(0.2e1)
t35 = math.sqrt(p_a_zeta_threshold)
t37 = jnp.where(0.1e1 <= p_a_zeta_threshold, t35 * p_a_zeta_threshold, 1)
res = -0.1925e0 + (0.863136e-1 * t5 + 0.572384e-1 * t9 + 0.3362975e-2 * t15) * t26 - 0.4e1 / 0.3e1 * (t29 - 0.1e1) * t31 * t2 * t3 * (t37 - 0.1e1)
