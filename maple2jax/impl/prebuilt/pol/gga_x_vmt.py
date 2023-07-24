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
t30 = params_a_mu * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t35 = t30 * t34
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t40 = 0.1e1 / t38 / t36
t42 = params_a_alpha * t29
t44 = t34 * s0 * t40
t47 = math.exp(-t42 * t44 / 0.24e2)
t60 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t35 * s0 * t40 * t47 / (0.1e1 + t30 * t44 / 0.24e2) / 0.24e2))
t62 = lax_cond(t10, t15, -t17)
t63 = lax_cond(t14, t11, t62)
t64 = 0.1e1 + t63
t66 = t64 ** (0.1e1 / 0.3e1)
t68 = lax_cond(t64 <= p_a_zeta_threshold, t23, t66 * t64)
t70 = r1 ** 2
t71 = r1 ** (0.1e1 / 0.3e1)
t72 = t71 ** 2
t74 = 0.1e1 / t72 / t70
t77 = t34 * s2 * t74
t80 = math.exp(-t42 * t77 / 0.24e2)
t93 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 + t35 * s2 * t74 * t80 / (0.1e1 + t30 * t77 / 0.24e2) / 0.24e2))
res = t60 + t93
