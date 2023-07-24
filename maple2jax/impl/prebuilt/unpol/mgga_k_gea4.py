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
t30 = t25 / t28
t31 = 2 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = r0 ** 2
t46 = t25 ** 2
t49 = t46 / t27 / t26
t50 = l0 ** 2
t59 = t34 ** 2
t66 = s0 ** 2
t78 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (0.1e1 + 0.5e1 / 0.648e3 * t30 * s0 * t32 / t23 / t34 + 0.5e1 / 0.54e2 * t30 * l0 * t32 / t23 / r0 + t49 * t50 * t31 / t22 / t34 / r0 / 0.2916e4 - t49 * s0 * t31 / t22 / t59 * l0 / 0.2592e4 + t49 * t66 * t31 / t22 / t59 / r0 / 0.8748e4))
res = 0.2e1 * t78
