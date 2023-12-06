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
t30 = r0 ** 2
t33 = math.sqrt(s0)
t34 = math.sqrt(r0)
t37 = t33 / t34 / r0
t38 = math.asinh(t37)
t50 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t28 * (0.1e1 + 0.26250000000000000000e-2 * t2 * s0 / t30 / r0 / (0.1e1 + 0.56e-1 * t37 * t38)))
t52 = lax_cond(t8, t13, -t15)
t53 = lax_cond(t12, t9, t52)
t54 = 0.1e1 + t53
t56 = math.sqrt(t54)
t58 = lax_cond(t54 <= p_a_zeta_threshold, t21, t56 * t54)
t61 = r1 ** 2
t64 = math.sqrt(s2)
t65 = math.sqrt(r1)
t68 = t64 / t65 / r1
t69 = math.asinh(t68)
t81 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t58 * t28 * (0.1e1 + 0.26250000000000000000e-2 * t2 * s2 / t61 / r1 / (0.1e1 + 0.56e-1 * t68 * t69)))
res = t50 + t81
