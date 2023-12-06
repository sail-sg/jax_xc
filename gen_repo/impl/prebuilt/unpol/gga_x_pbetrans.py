t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = math.pi ** 2
t22 = t21 ** (0.1e1 / 0.3e1)
t24 = 6 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t28 = math.sqrt(s0)
t29 = 2 ** (0.1e1 / 0.3e1)
t39 = math.exp(-0.2e1 * t3 * t22 * (t25 / t22 * t28 * t29 / t19 / r0 / 0.12e2 - 0.3e1))
t42 = 0.413e0 / (0.1e1 + t39)
t43 = 0.1227e1 - t42
t44 = t22 ** 2
t47 = t29 ** 2
t49 = r0 ** 2
t50 = t19 ** 2
t65 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t43 * (0.1e1 - t43 / (0.1227e1 - t42 + 0.91249999999999999998e-2 * t24 / t44 * s0 * t47 / t50 / t49))))
res = 0.2e1 * t65
