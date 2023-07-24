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
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = 0.1e1 / t33
t35 = params_a_B * t29 * t34
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t40 = 0.1e1 / t38 / t36
t41 = s0 * t40
t42 = params_a_C * t29
t53 = params_a_D * t29 * t34
t54 = t29 ** 2
t55 = params_a_E * t54
t57 = 0.1e1 / t32 / t31
t58 = s0 ** 2
t60 = t36 ** 2
t76 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (params_a_A + t35 * t41 / (0.1e1 + t42 * t34 * s0 * t40 / 0.24e2) / 0.24e2 - t53 * t41 / (0.1e1 + t55 * t57 * t58 / t37 / t60 / r0 / 0.576e3) / 0.24e2))
t78 = lax_cond(t10, t15, -t17)
t79 = lax_cond(t14, t11, t78)
t80 = 0.1e1 + t79
t82 = t80 ** (0.1e1 / 0.3e1)
t84 = lax_cond(t80 <= p_a_zeta_threshold, t23, t82 * t80)
t86 = r1 ** 2
t87 = r1 ** (0.1e1 / 0.3e1)
t88 = t87 ** 2
t90 = 0.1e1 / t88 / t86
t91 = s2 * t90
t101 = s2 ** 2
t103 = t86 ** 2
t119 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t84 * t27 * (params_a_A + t35 * t91 / (0.1e1 + t42 * t34 * s2 * t90 / 0.24e2) / 0.24e2 - t53 * t91 / (0.1e1 + t55 * t57 * t101 / t87 / t103 / r1 / 0.576e3) / 0.24e2))
res = t76 + t119
