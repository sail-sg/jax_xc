t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = jnp.where(t8, -t9, 0)
t12 = jnp.where(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = jnp.where(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = 2 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t28 = r0 ** 2
t31 = math.sqrt(s0)
t32 = t31 * t25
t34 = 0.1e1 / t22 / r0
t36 = math.asinh(t32 * t34)
t45 = 4 ** (0.1e1 / 0.3e1)
t59 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + 0.55e-2 * s0 * t26 / t23 / t28 / (0.1e1 + 0.253e-1 * t32 * t34 * t36) - 0.72e-1 * t32 * t34 / (0.2e1 * t45 * t31 * t25 * t34 + 0.1e1)))
res = 0.2e1 * t59
