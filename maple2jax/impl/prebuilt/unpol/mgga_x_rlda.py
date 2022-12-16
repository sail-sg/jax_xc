t3 = math.pi ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = 0.1e1 <= p_a_zeta_threshold
t6 = p_a_zeta_threshold - 0.1e1
t8 = jnp.where(t5, -t6, 0)
t9 = jnp.where(t5, t6, t8)
t10 = 0.1e1 + t9
t12 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t14 = t10 ** (0.1e1 / 0.3e1)
t16 = jnp.where(t10 <= p_a_zeta_threshold, t12 * p_a_zeta_threshold, t14 * t10)
t18 = r0 ** (0.1e1 / 0.3e1)
t21 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t24 = 4 ** (0.1e1 / 0.3e1)
t25 = 2 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t28 = t18 ** 2
t30 = 0.1e1 / t28 / r0
t42 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.15e2 / 0.16e2 * t4 * t16 * t18 * params_a_prefactor / t21 * t24 / (0.2e1 * tau0 * t26 * t30 - l0 * t26 * t30 / 0.4e1))
res = 0.2e1 * t42
