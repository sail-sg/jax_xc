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
t29 = 0.1e1 / math.pi
t30 = 6 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = math.pi ** 2
t34 = t33 ** (0.1e1 / 0.3e1)
t36 = t2 ** 2
t38 = t29 ** (0.1e1 / 0.3e1)
t41 = t29 * t31 / t34 * t36 / t38
t42 = 4 ** (0.1e1 / 0.3e1)
t44 = r0 ** 2
t45 = r0 ** (0.1e1 / 0.3e1)
t46 = t45 ** 2
t48 = 0.1e1 / t46 / t44
t49 = math.sqrt(s0)
t53 = (t49 / t45 / r0) ** params_a_alpha
t69 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.7e1 / 0.11664e5 * t41 * t42 * s0 * t48 * (params_a_a1 * t53 + 0.1e1) / (params_a_b1 * s0 * t48 + 0.1e1)))
t71 = lax_cond(t10, t15, -t17)
t72 = lax_cond(t14, t11, t71)
t73 = 0.1e1 + t72
t75 = t73 ** (0.1e1 / 0.3e1)
t77 = lax_cond(t73 <= p_a_zeta_threshold, t23, t75 * t73)
t80 = r1 ** 2
t81 = r1 ** (0.1e1 / 0.3e1)
t82 = t81 ** 2
t84 = 0.1e1 / t82 / t80
t85 = math.sqrt(s2)
t89 = (t85 / t81 / r1) ** params_a_alpha
t105 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t77 * t27 * (0.1e1 + 0.7e1 / 0.11664e5 * t41 * t42 * s2 * t84 * (params_a_a1 * t89 + 0.1e1) / (params_a_b1 * s2 * t84 + 0.1e1)))
res = t69 + t105
