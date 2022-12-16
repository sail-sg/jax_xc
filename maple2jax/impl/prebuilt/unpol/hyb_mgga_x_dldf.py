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
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t20 ** 2
t42 = t21 ** 2
t44 = 0.3e1 / 0.10e2 * t42 * t24
t48 = tau0 * t28 / t31 / r0
t49 = t44 - t48
t50 = t44 + t48
t54 = t49 ** 2
t55 = t50 ** 2
t64 = t54 ** 2
t65 = t55 ** 2
t73 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.14459516250000000000e0 * t3 / t4 * t18 * t20 * (0.58827323e1 - 0.2384107471346329e2 / (0.48827323e1 + 0.14629700000000000000e-1 * t21 / t24 * s0 * t28 / t31 / t30)) * (0.1e1 - 0.1637571e0 * t49 / t50 - 0.1880028e0 * t54 / t55 - 0.4490609e0 * t54 * t49 / t55 / t50 - 0.82359e-2 * t64 / t65))
res = 0.2e1 * t73
