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
t22 = t21 ** 2
t24 = math.pi ** 2
t25 = t24 ** (0.1e1 / 0.3e1)
t26 = 0.1e1 / t25
t28 = math.sqrt(s0)
t29 = 2 ** (0.1e1 / 0.3e1)
t30 = t28 * t29
t32 = 0.1e1 / t19 / r0
t38 = math.log(0.1e1 + t22 * t26 * t30 * t32 / 0.12e2)
t46 = math.log(0.1e1 + t38)
t55 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params_a_B1 * t22 * t26 * t30 * t32 * t38 / 0.12e2 + params_a_B2 * t22 * t26 * t30 * t32 * t46 / 0.12e2))
res = 0.2e1 * t55
