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
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t31 = r0 ** 2
t32 = t19 ** 2
t35 = math.sqrt(s0)
t36 = t35 * t28
t38 = 0.1e1 / t19 / r0
t40 = math.asinh(t36 * t38)
t44 = 0.1e1 + 0.252e-1 * t36 * t38 * t40
t47 = t44 ** 2
t59 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.109878e1 + 0.93333333333333333332e-3 * t21 / t23 * t26 * s0 * t29 / t32 / t31 * (-0.251173e1 / t44 + 0.37198333333333333333e1 / t47)))
res = 0.2e1 * t59
