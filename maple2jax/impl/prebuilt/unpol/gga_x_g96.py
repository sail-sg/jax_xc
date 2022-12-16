t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = t3 ** 2
t23 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t26 = 4 ** (0.1e1 / 0.3e1)
t27 = math.sqrt(s0)
t28 = 2 ** (0.1e1 / 0.3e1)
t32 = t27 * t28 / t19 / r0
t33 = math.sqrt(t32)
t42 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.1233e4 * t21 / t23 * t26 * t33 * t32))
res = 0.2e1 * t42
