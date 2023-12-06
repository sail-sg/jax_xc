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
t28 = math.sqrt(t4)
t29 = r0 ** 2
t32 = s0 / t29 / r0
t42 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t26 * t28 * (0.1e1 + 0.2105e-2 * t32) / (0.1e1 + 0.119e-3 * t32))
t44 = lax_cond(t8, t13, -t15)
t45 = lax_cond(t12, t9, t44)
t46 = 0.1e1 + t45
t48 = math.sqrt(t46)
t50 = lax_cond(t46 <= p_a_zeta_threshold, t21, t48 * t46)
t53 = r1 ** 2
t56 = s2 / t53 / r1
t66 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t50 * t26 * t28 * (0.1e1 + 0.2105e-2 * t56) / (0.1e1 + 0.119e-3 * t56))
res = t42 + t66
