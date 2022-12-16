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
t27 = math.pi ** 2
t28 = t27 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = 0.1e1 / t29
t32 = 2 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = s0 * t33
t35 = r0 ** 2
t37 = 0.1e1 / t23 / t35
t43 = 0.1e1 + params_a_a * t25 * t30 * t34 * t37 / 0.24e2
t49 = t25 ** 2
t54 = s0 ** 2
t56 = t35 ** 2
t60 = t43 ** 2
t66 = t27 ** 2
t70 = t56 ** 2
t82 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + params_a_c1 * t25 * t30 * t34 * t37 / t43 / 0.24e2 + params_a_c2 * t49 / t28 / t27 * t54 * t32 / t22 / t56 / r0 / t60 / 0.288e3 + params_a_c3 / t66 * t54 * s0 / t70 / t60 / t43 / 0.576e3))
res = 0.2e1 * t82
