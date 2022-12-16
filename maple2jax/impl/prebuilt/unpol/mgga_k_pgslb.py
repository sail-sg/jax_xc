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
t25 = 6 ** (0.1e1 / 0.3e1)
t26 = math.pi ** 2
t27 = t26 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t29 = 0.1e1 / t28
t31 = 2 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = r0 ** 2
t37 = s0 * t32 / t23 / t34
t44 = math.exp(-params_a_pgslb_mu * t25 * t29 * t37 / 0.24e2)
t45 = t25 ** 2
t50 = l0 ** 2
t62 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.5e1 / 0.72e2 * t25 * t29 * t37 + t44 + params_a_pgslb_beta * t45 / t27 / t26 * t50 * t31 / t22 / t34 / r0 / 0.288e3))
res = 0.2e1 * t62
