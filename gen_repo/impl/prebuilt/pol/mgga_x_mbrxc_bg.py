t2 = r0 + r1
t3 = 0.1e1 / t2
t6 = 0.2e1 * r0 * t3 <= p_a_zeta_threshold
t7 = p_a_zeta_threshold - 0.1e1
t10 = 0.2e1 * r1 * t3 <= p_a_zeta_threshold
t11 = -t7
t13 = (r0 - r1) * t3
t14 = lax_cond(t10, t11, t13)
t15 = lax_cond(t6, t7, t14)
t16 = 0.1e1 + t15
t18 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t19 = t18 * p_a_zeta_threshold
t20 = t16 ** (0.1e1 / 0.3e1)
t22 = lax_cond(t16 <= p_a_zeta_threshold, t19, t20 * t16)
t23 = t2 ** (0.1e1 / 0.3e1)
t25 = 32 ** (0.1e1 / 0.3e1)
t27 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t29 = t25 / t27
t31 = 4 ** (0.1e1 / 0.3e1)
t32 = r0 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t38 = 6 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t40 = math.pi ** 2
t41 = t40 ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t44 = 0.3e1 / 0.10e2 * t39 * t42
t45 = r0 ** 2
t50 = s0 ** 2
t51 = t45 ** 2
t57 = 0.149492e0 * tau0 / t33 / r0 - t44 + 0.147e0 * s0 / t33 / t45 + 0.32e-2 * t50 / t32 / t51 / r0
t58 = abs(t57)
t61 = lax_cond(0.0e0 < t57, 0.50e-12, -0.50e-12)
t62 = lax_cond(t58 < 0.50e-12, t61, t57)
t63 = mbrxc_x(t62)
t65 = math.exp(t63 / 0.3e1)
t67 = math.exp(-t63)
t68 = t63 ** 2
t76 = (0.1e1 + t63) ** (0.1e1 / 0.3e1)
t82 = lax_cond(r0 <= p_a_dens_threshold, 0, -t22 * t23 * t29 * t31 * t65 * (0.8e1 - t67 * (t68 + 0.5e1 * t63 + 0.8e1)) / t63 / t76 / 0.64e2)
t84 = lax_cond(t6, t11, -t13)
t85 = lax_cond(t10, t7, t84)
t86 = 0.1e1 + t85
t88 = t86 ** (0.1e1 / 0.3e1)
t90 = lax_cond(t86 <= p_a_zeta_threshold, t19, t88 * t86)
t93 = r1 ** (0.1e1 / 0.3e1)
t94 = t93 ** 2
t99 = r1 ** 2
t104 = s2 ** 2
t105 = t99 ** 2
t111 = 0.149492e0 * tau1 / t94 / r1 - t44 + 0.147e0 * s2 / t94 / t99 + 0.32e-2 * t104 / t93 / t105 / r1
t112 = abs(t111)
t115 = lax_cond(0.0e0 < t111, 0.50e-12, -0.50e-12)
t116 = lax_cond(t112 < 0.50e-12, t115, t111)
t117 = mbrxc_x(t116)
t119 = math.exp(t117 / 0.3e1)
t121 = math.exp(-t117)
t122 = t117 ** 2
t130 = (0.1e1 + t117) ** (0.1e1 / 0.3e1)
t136 = lax_cond(r1 <= p_a_dens_threshold, 0, -t90 * t23 * t29 * t31 * t119 * (0.8e1 - t121 * (t122 + 0.5e1 * t117 + 0.8e1)) / t117 / t130 / 0.64e2)
res = t82 + t136
