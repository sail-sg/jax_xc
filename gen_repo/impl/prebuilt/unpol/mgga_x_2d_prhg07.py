t3 = 0.1e1 <= p_a_zeta_threshold
t4 = p_a_zeta_threshold - 0.1e1
t6 = lax_cond(t3, -t4, 0)
t7 = lax_cond(t3, t4, t6)
t8 = 0.1e1 + t7
t10 = math.sqrt(p_a_zeta_threshold)
t12 = math.sqrt(t8)
t14 = lax_cond(t8 <= p_a_zeta_threshold, t10 * p_a_zeta_threshold, t12 * t8)
t16 = math.sqrt(0.2e1)
t17 = math.sqrt(r0)
t19 = r0 ** 2
t20 = 0.1e1 / t19
t31 = (l0 * t20 / 0.2e1 - 0.2e1 * tau0 * t20 + s0 / t19 / r0 / 0.4e1) / math.pi
t33 = lax_cond(-0.9999999999e0 < t31, t31, -0.9999999999e0)
t34 = math.exp(-1)
t36 = scipy.special.lambertw(t33 * t34)
t39 = scipy.special.i0(t36 / 0.2e1 + 0.1e1 / 0.2e1)
t43 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -math.pi * t14 * t16 * t17 * t39 / 0.8e1)
res = 0.2e1 * t43
