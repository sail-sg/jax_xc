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
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t31 = r0 ** 2
t32 = t19 ** 2
t38 = t21 ** 2
t43 = s0 ** 2
t45 = t31 ** 2
t52 = t23 ** 2
t56 = t45 ** 2
t62 = (0.1e1 + params_a_aa * t21 / t25 * s0 * t29 / t32 / t31 / 0.24e2 + params_a_bb * t38 / t24 / t23 * t43 * t28 / t19 / t45 / r0 / 0.288e3 + params_a_cc / t52 * t43 * s0 / t56 / 0.576e3) ** (0.1e1 / 0.15e2)
t66 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t62)
res = 0.2e1 * t66
