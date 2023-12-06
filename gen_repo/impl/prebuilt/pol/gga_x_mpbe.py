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
t35 = params_a_c1 * t29 * t34
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t40 = 0.1e1 / t38 / t36
t42 = params_a_a * t29
t47 = 0.1e1 + t42 * t34 * s0 * t40 / 0.24e2
t52 = t29 ** 2
t56 = params_a_c2 * t52 / t32 / t31
t57 = s0 ** 2
t58 = t36 ** 2
t63 = t47 ** 2
t68 = t31 ** 2
t70 = params_a_c3 / t68
t72 = t58 ** 2
t84 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t35 * s0 * t40 / t47 / 0.24e2 + t56 * t57 / t37 / t58 / r0 / t63 / 0.576e3 + t70 * t57 * s0 / t72 / t63 / t47 / 0.2304e4))
t86 = lax_cond(t10, t15, -t17)
t87 = lax_cond(t14, t11, t86)
t88 = 0.1e1 + t87
t90 = t88 ** (0.1e1 / 0.3e1)
t92 = lax_cond(t88 <= p_a_zeta_threshold, t23, t90 * t88)
t94 = r1 ** 2
t95 = r1 ** (0.1e1 / 0.3e1)
t96 = t95 ** 2
t98 = 0.1e1 / t96 / t94
t104 = 0.1e1 + t42 * t34 * s2 * t98 / 0.24e2
t109 = s2 ** 2
t110 = t94 ** 2
t115 = t104 ** 2
t121 = t110 ** 2
t133 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t92 * t27 * (0.1e1 + t35 * s2 * t98 / t104 / 0.24e2 + t56 * t109 / t95 / t110 / r1 / t115 / 0.576e3 + t70 * t109 * s2 / t121 / t115 / t104 / 0.2304e4))
res = t84 + t133
