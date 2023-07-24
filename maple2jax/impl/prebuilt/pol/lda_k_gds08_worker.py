t3 = r0 + r1
t4 = 0.1e1 / t3
t5 = (r0 - r1) * t4
t7 = 0.1e1 + t5 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = 0.1e1 - t5 <= p_a_zeta_threshold
t11 = -t8
t12 = lax_cond(t10, t11, t5)
t13 = lax_cond(t7, t8, t12)
t17 = 0.2e1 * r0 * t4 <= p_a_zeta_threshold
t20 = 0.2e1 * r1 * t4 <= p_a_zeta_threshold
t21 = lax_cond(t20, t11, t5)
t22 = lax_cond(t17, t8, t21)
t25 = math.log((0.1e1 + t22) * t3)
t27 = t25 ** 2
t32 = lax_cond(r0 <= p_a_dens_threshold, 0, (0.1e1 + t13) * (params_a_B * t25 + params_a_C * t27 + params_a_A) / 0.2e1)
t34 = lax_cond(t7, t11, -t5)
t35 = lax_cond(t10, t8, t34)
t37 = lax_cond(t17, t11, -t5)
t38 = lax_cond(t20, t8, t37)
t41 = math.log((0.1e1 + t38) * t3)
t43 = t41 ** 2
t48 = lax_cond(r1 <= p_a_dens_threshold, 0, (0.1e1 + t35) * (params_a_B * t41 + params_a_C * t43 + params_a_A) / 0.2e1)
res = t32 + t48
