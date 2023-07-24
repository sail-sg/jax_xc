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
t26 = math.pi ** 2
t27 = t26 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t31 = 2 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = r0 ** 2
t40 = math.exp(-0.83254166666666666664e1 * t25 / t28 * s0 * t32 / t23 / t34)
t42 = t25 ** 2
t46 = s0 ** 2
t48 = t34 ** 2
t55 = math.exp(-0.15095833333333333333e-1 * t42 / t27 / t26 * t46 * t31 / t22 / t48 / r0)
t61 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.20788e1 - 0.8524e0 * t40 - 0.12264e1 * t55))
res = 0.2e1 * t61
