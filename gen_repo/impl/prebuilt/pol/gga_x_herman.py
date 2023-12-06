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
t29 = t2 ** 2
t31 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t33 = t29 / t31
t34 = 4 ** (0.1e1 / 0.3e1)
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t48 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.66666666666666666668e-3 * t33 * t34 * s0 / t38 / t36))
t50 = lax_cond(t10, t15, -t17)
t51 = lax_cond(t14, t11, t50)
t52 = 0.1e1 + t51
t54 = t52 ** (0.1e1 / 0.3e1)
t56 = lax_cond(t52 <= p_a_zeta_threshold, t23, t54 * t52)
t59 = r1 ** 2
t60 = r1 ** (0.1e1 / 0.3e1)
t61 = t60 ** 2
t71 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t56 * t27 * (0.1e1 + 0.66666666666666666668e-3 * t33 * t34 * s2 / t61 / t59))
res = t48 + t71
