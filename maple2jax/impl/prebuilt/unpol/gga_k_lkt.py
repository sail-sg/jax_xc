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
t25 = 6 ** (0.1e1 / 0.3e1)
t26 = t25 ** 2
t27 = math.pi ** 2
t28 = t27 ** (0.1e1 / 0.3e1)
t31 = math.sqrt(s0)
t32 = 2 ** (0.1e1 / 0.3e1)
t38 = t26 / t28 * t31 * t32 / t22 / r0 / 0.12e2
t40 = lax_cond(t38 < 0.200e3, t38, 200)
t42 = math.cosh(params_a_a * t40)
t44 = t28 ** 2
t47 = t32 ** 2
t49 = r0 ** 2
t59 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 / t42 + 0.5e1 / 0.72e2 * t25 / t44 * s0 * t47 / t23 / t49))
res = 0.2e1 * t59
