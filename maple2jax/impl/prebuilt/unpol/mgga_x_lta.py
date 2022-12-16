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
t21 = 2 ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t24 = t19 ** 2
t27 = 6 ** (0.1e1 / 0.3e1)
t29 = math.pi ** 2
t30 = t29 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t37 = (0.5e1 / 0.9e1 * tau0 * t22 / t24 / r0 * t27 / t31) ** (0.4e1 / 0.5e1 * params_a_ltafrac)
t41 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t37)
res = 0.2e1 * t41
