t2 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = jnp.where(0.1e1 <= p_a_zeta_threshold, t3, 1)
t5 = t4 ** 2
t7 = 3 ** (0.1e1 / 0.3e1)
t9 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t11 = 4 ** (0.1e1 / 0.3e1)
t12 = t11 ** 2
t13 = r0 ** (0.1e1 / 0.3e1)
t19 = math.atan(0.4888270e1 + 0.79425925000000000000e0 * t7 * t9 * t12 / t13)
t23 = t7 ** 2
res = t5 * t4 * (-0.655868e0 * t19 + 0.897889e0) * t23 / t9 * t11 * t13 / 0.3e1
