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
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = 2 ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t24 = t20 ** 2
t27 = tau0 * t22 / t24 / r0
t29 = r0 ** 2
t34 = t27 - s0 * t22 / t24 / t29 / 0.8e1
t35 = 6 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t37 = math.pi ** 2
t38 = t37 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t42 = t27 - 0.3e1 / 0.10e2 * t36 * t39
t47 = t34 ** 2
t49 = t42 ** 2
t53 = (0.1e1 + params_a_e1 * t47 / t49) ** 2
t54 = t47 ** 2
t56 = t49 ** 2
t60 = (t53 + params_a_c1 * t54 / t56) ** (0.1e1 / 0.4e1)
t69 = s0 ** 2
t71 = t29 ** 2
t79 = (0.1e1 + params_a_b * t36 / t38 / t37 * t69 * t21 / t20 / t71 / r0 / 0.288e3) ** (0.1e1 / 0.8e1)
t84 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * (0.1e1 + params_a_k0 * (0.1e1 - t34 / t42) / t60) / t79)
res = 0.2e1 * t84
