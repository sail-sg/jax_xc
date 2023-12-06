t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = lax_cond(t8, -t9, 0)
t12 = lax_cond(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = lax_cond(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = 2 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t28 = r0 ** 2
t33 = math.sqrt(s0)
t40 = 6 ** (0.1e1 / 0.3e1)
t42 = math.pi ** 2
t43 = t42 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t52 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + 0.5e1 / 0.9e1 * (s0 * t26 / t23 / t28 / 0.72e2 + 0.677e-2 * t26 * t33 / t22 / r0) * t40 / t44))
res = 0.2e1 * t52
