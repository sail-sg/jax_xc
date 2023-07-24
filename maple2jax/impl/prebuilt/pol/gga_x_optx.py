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
t29 = params_a_gamma ** 2
t30 = params_a_b * t29
t31 = s0 ** 2
t32 = r0 ** 2
t33 = t32 ** 2
t35 = r0 ** (0.1e1 / 0.3e1)
t40 = t35 ** 2
t45 = (0.1e1 + params_a_gamma * s0 / t40 / t32) ** 2
t53 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (params_a_a + t30 * t31 / t35 / t33 / r0 / t45))
t55 = lax_cond(t10, t15, -t17)
t56 = lax_cond(t14, t11, t55)
t57 = 0.1e1 + t56
t59 = t57 ** (0.1e1 / 0.3e1)
t61 = lax_cond(t57 <= p_a_zeta_threshold, t23, t59 * t57)
t63 = s2 ** 2
t64 = r1 ** 2
t65 = t64 ** 2
t67 = r1 ** (0.1e1 / 0.3e1)
t72 = t67 ** 2
t77 = (0.1e1 + params_a_gamma * s2 / t72 / t64) ** 2
t85 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t61 * t27 * (params_a_a + t30 * t63 / t67 / t65 / r1 / t77))
res = t53 + t85
