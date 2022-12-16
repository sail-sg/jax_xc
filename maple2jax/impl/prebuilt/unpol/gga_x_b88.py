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
t24 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t26 = 4 ** (0.1e1 / 0.3e1)
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t32 = r0 ** 2
t33 = t19 ** 2
t37 = math.sqrt(s0)
t40 = 0.1e1 / t19 / r0
t44 = math.asinh(t37 * t29 * t40)
t57 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.9e1 * params_a_beta * t21 / t24 * t26 * s0 * t30 / t33 / t32 / (params_a_gamma * params_a_beta * t37 * t29 * t40 * t44 + 0.1e1)))
res = 0.2e1 * t57
