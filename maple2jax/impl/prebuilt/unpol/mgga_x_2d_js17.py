t3 = math.sqrt(math.pi)
t5 = 0.1e1 <= p_a_zeta_threshold
t6 = p_a_zeta_threshold - 0.1e1
t8 = jnp.where(t5, -t6, 0)
t9 = jnp.where(t5, t6, t8)
t10 = 0.1e1 + t9
t12 = math.sqrt(p_a_zeta_threshold)
t14 = math.sqrt(t10)
t16 = jnp.where(t10 <= p_a_zeta_threshold, t12 * p_a_zeta_threshold, t14 * t10)
t18 = math.sqrt(0.2e1)
t19 = math.sqrt(r0)
t21 = 0.1e1 / math.pi
t23 = r0 ** 2
t26 = t21 * s0 / t23 / r0
t28 = math.pi ** 2
t30 = s0 ** 2
t32 = t23 ** 2
t37 = 0.1e1 + 0.25920000000000000000e1 * t26 + 0.24883200000000000000e-1 / t28 * t30 / t32 / t23
t38 = t37 ** (0.1e1 / 0.15e2)
t49 = t37 ** (0.1e1 / 0.5e1)
t57 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 / t3 * t16 * t18 * t19 * (0.1e1 / t38 + 0.2e1 / 0.5e1 * (0.1e1 + 0.17554285714285714286e0 * t26 + (-0.1544000000000000000e0 * tau0 / t23 - 0.36912000000000000000e1 * math.pi) * t21 / 0.4e1) / t49))
res = 0.2e1 * t57
