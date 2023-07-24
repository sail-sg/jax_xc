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
t20 = t19 + 0.1e1
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t28 = t6 ** (0.1e1 / 0.3e1)
t29 = t3 ** 2
t30 = t2 * t29
t31 = 2 ** (0.1e1 / 0.3e1)
t33 = r0 ** 2
t34 = r0 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t41 = math.pi ** 2
t42 = t2 ** 2
t43 = t42 * t3
t44 = t31 ** 2
t45 = math.sqrt(s0)
t50 = t43 * t44 * t45 / t34 / r0
t53 = math.log(t50 / 0.27e2 + 0.1e1)
t65 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.2e1 / 0.81e2 * t30 * t31 * s0 / t35 / t33 + t41 * t53) / (t50 / 0.9e1 + t41) / t53)
t67 = lax_cond(t10, t15, -t17)
t68 = lax_cond(t14, t11, t67)
t69 = t68 + 0.1e1
t71 = t69 ** (0.1e1 / 0.3e1)
t73 = lax_cond(t69 <= p_a_zeta_threshold, t23, t71 * t69)
t76 = r1 ** 2
t77 = r1 ** (0.1e1 / 0.3e1)
t78 = t77 ** 2
t84 = math.sqrt(s2)
t89 = t43 * t44 * t84 / t77 / r1
t92 = math.log(t89 / 0.27e2 + 0.1e1)
t104 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t73 * t28 * (0.2e1 / 0.81e2 * t30 * t31 * s2 / t78 / t76 + t41 * t92) / (t89 / 0.9e1 + t41) / t92)
res = t65 + t104
