t2 = math.sqrt(math.pi)
t3 = 0.1e1 / t2
t4 = r0 + r1
t5 = 0.1e1 / t4
t8 = 0.2e1 * r0 * t5 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t12 = 0.2e1 * r1 * t5 <= p_a_zeta_threshold
t13 = -t9
t15 = (r0 - r1) * t5
t16 = lax_cond(t12, t13, t15)
t17 = lax_cond(t8, t9, t16)
t18 = 0.1e1 + t17
t20 = math.sqrt(p_a_zeta_threshold)
t21 = t20 * p_a_zeta_threshold
t22 = math.sqrt(t18)
t24 = lax_cond(t18 <= p_a_zeta_threshold, t21, t22 * t18)
t26 = math.sqrt(0.2e1)
t27 = math.sqrt(t4)
t28 = t26 * t27
t29 = 0.1e1 / math.pi
t31 = r0 ** 2
t43 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t28 * (0.14604e1 - 0.21196816e0 / (0.4604e0 + 0.22159179687500000000e-1 * t29 * s0 / t31 / r0)))
t45 = lax_cond(t8, t13, -t15)
t46 = lax_cond(t12, t9, t45)
t47 = 0.1e1 + t46
t49 = math.sqrt(t47)
t51 = lax_cond(t47 <= p_a_zeta_threshold, t21, t49 * t47)
t54 = r1 ** 2
t66 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t51 * t28 * (0.14604e1 - 0.21196816e0 / (0.4604e0 + 0.22159179687500000000e-1 * t29 * s2 / t54 / r1)))
res = t43 + t66
