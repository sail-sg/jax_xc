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
t25 = 0.1e1 / t24
t26 = t21 * t25
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t34 = s0 * t28 / t31 / t30
t35 = t26 * t34
t44 = t26 * tau0 * t28 / t31 / r0 / 0.4e1 - 0.9e1 / 0.20e2 - t35 / 0.288e3
t45 = t44 ** 2
t51 = t21 ** 2
t55 = s0 ** 2
t57 = t30 ** 2
t71 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1804e1 - 0.646416e0 / (0.804e0 + 0.5e1 / 0.972e3 * t35 + 0.146e3 / 0.2025e4 * t45 - 0.73e2 / 0.9720e4 * t44 * t21 * t25 * t34 + 0.45818468001825619316e-3 * t51 / t23 / t22 * t55 * t27 / t19 / t57 / r0)))
res = 0.2e1 * t71
