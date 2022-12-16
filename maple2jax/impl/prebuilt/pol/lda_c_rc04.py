t2 = r0 + r1
t4 = (r0 - r1) / t2
t5 = 0.1e1 + t4
t7 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t8 = t7 ** 2
t9 = t5 ** (0.1e1 / 0.3e1)
t10 = t9 ** 2
t11 = jnp.where(t5 <= p_a_zeta_threshold, t8, t10)
t12 = 0.1e1 - t4
t14 = t12 ** (0.1e1 / 0.3e1)
t15 = t14 ** 2
t16 = jnp.where(t12 <= p_a_zeta_threshold, t8, t15)
t18 = t11 / 0.2e1 + t16 / 0.2e1
t19 = t18 ** 2
t21 = 3 ** (0.1e1 / 0.3e1)
t23 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t25 = 4 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t27 = t2 ** (0.1e1 / 0.3e1)
t33 = math.atan(0.4888270e1 + 0.79425925000000000000e0 * t21 * t23 * t26 / t27)
t37 = t21 ** 2
res = t19 * t18 * (-0.655868e0 * t33 + 0.897889e0) * t37 / t23 * t25 * t27 / 0.3e1
