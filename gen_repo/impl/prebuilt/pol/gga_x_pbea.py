t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = lax_cond(t14, t15, t17)
t19 = lax_cond(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = r0 ** 2
t30 = r0 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t37 = (0.1e1 + 0.86399408095363255118e-2 * s0 / t31 / t29) ** (-0.52e0)
t43 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.18040e1 - 0.8040e0 * t37))
t45 = lax_cond(t10, t15, -t17)
t46 = lax_cond(t14, t11, t45)
t47 = 0.1e1 + t46
t49 = t47 ** (0.1e1 / 0.3e1)
t51 = lax_cond(t47 <= p_a_zeta_threshold, t23, t49 * t47)
t53 = r1 ** 2
t54 = r1 ** (0.1e1 / 0.3e1)
t55 = t54 ** 2
t61 = (0.1e1 + 0.86399408095363255118e-2 * s2 / t55 / t53) ** (-0.52e0)
t67 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t51 * t27 * (0.18040e1 - 0.8040e0 * t61))
res = t43 + t67
