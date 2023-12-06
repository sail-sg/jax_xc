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
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t32 = math.pi ** 2
t33 = t32 ** (0.1e1 / 0.3e1)
t34 = 0.1e1 / t33
t35 = params_a_alphaoAx * t30 * t34
t36 = math.sqrt(s0)
t37 = r0 ** (0.1e1 / 0.3e1)
t40 = t36 / t37 / r0
t41 = t30 * t34
t45 = math.log(0.1e1 + t41 * t40 / 0.12e2)
t57 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - t35 * t40 * t45 / (params_a_c * t45 + 0.1e1) / 0.12e2))
t59 = lax_cond(t10, t15, -t17)
t60 = lax_cond(t14, t11, t59)
t61 = 0.1e1 + t60
t63 = t61 ** (0.1e1 / 0.3e1)
t65 = lax_cond(t61 <= p_a_zeta_threshold, t23, t63 * t61)
t67 = math.sqrt(s2)
t68 = r1 ** (0.1e1 / 0.3e1)
t71 = t67 / t68 / r1
t75 = math.log(0.1e1 + t41 * t71 / 0.12e2)
t87 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t65 * t27 * (0.1e1 - t35 * t71 * t75 / (params_a_c * t75 + 0.1e1) / 0.12e2))
res = t57 + t87
