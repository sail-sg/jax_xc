t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t5 = 0.1e1 / t4
t8 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t10 = lax_cond(0.1e1 <= p_a_zeta_threshold, t8 * p_a_zeta_threshold, 1)
t11 = r0 ** (0.1e1 / 0.3e1)
t15 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 * t5 * t10 * t11)
t16 = 9 ** (0.1e1 / 0.3e1)
t17 = t16 ** 2
t18 = t4 ** 2
t21 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t22 = t21 ** 2
t25 = t11 ** 2
t30 = math.sqrt(0.1e1 + 0.17750451365686221606e-4 * t17 * t18 * t3 / t22 * t25)
t39 = t3 ** 2
t45 = math.asinh(0.24324508467583486202e-2 * t16 * t4 * t39 / t21 * t11)
t55 = (0.15226222180972388889e2 * t30 * t17 * t5 * t3 * t21 / t11 - 0.20865405771390201384e4 * t45 * t16 / t18 * t39 * t22 / t25) ** 2
res = 0.2e1 * t15 * (0.1e1 - 0.15e1 * t55)
