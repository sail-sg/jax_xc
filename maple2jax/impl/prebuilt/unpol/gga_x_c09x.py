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
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t26 = t21 / t24
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t33 = 0.1e1 / t31 / t30
t37 = t26 * s0 * t29 * t33
t39 = math.exp(-0.20125000000000000000e-2 * t37)
t44 = math.exp(-0.10062500000000000000e-2 * t37)
t50 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.2245e1 + 0.25708333333333333333e-2 * t26 * s0 * t29 * t33 * t39 - 0.1245e1 * t44))
res = 0.2e1 * t50
