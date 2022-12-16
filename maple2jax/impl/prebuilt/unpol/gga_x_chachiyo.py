t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = t11 + 0.1e1
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = t4 ** 2
t23 = r0 ** 2
t24 = t20 ** 2
t30 = math.pi ** 2
t31 = t3 ** 2
t33 = math.sqrt(s0)
t37 = t31 * t4 * t33 / t20 / r0
t40 = math.log(0.2e1 / 0.27e2 * t37 + 0.1e1)
t52 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.4e1 / 0.81e2 * t3 * t21 * s0 / t24 / t23 + t30 * t40) / (0.2e1 / 0.9e1 * t37 + t30) / t40)
res = 0.2e1 * t52
