t1 = 3 ** (0.1e1 / 0.3e1)
t2 = t1 ** 2
t4 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t7 = 4 ** (0.1e1 / 0.3e1)
t10 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t11 = t10 ** 2
t13 = jnp.where(0.1e1 <= p_a_zeta_threshold, t11 * p_a_zeta_threshold, 1)
t14 = r0 ** (0.1e1 / 0.3e1)
t15 = t14 ** 2
t20 = math.log(0.1e1 + 0.51020408163265306120e3 / t14)
res = 0.10790666666666666667e1 * t2 / t4 * t7 * t13 * t15 * (0.1e1 - 0.19600000000000000000e-2 * t14 * t20)
