t1 = 3 ** (0.1e1 / 0.3e1)
t3 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = t1 * t3 * t6
t8 = 2 ** (0.1e1 / 0.3e1)
t9 = t8 ** 2
t11 = r0 + r1
t13 = (r0 - r1) / t11
t14 = 0.1e1 + t13
t15 = t14 <= p_a_zeta_threshold
t16 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t17 = t16 * p_a_zeta_threshold
t18 = t14 ** (0.1e1 / 0.3e1)
t20 = lax_cond(t15, t17, t18 * t14)
t22 = t11 ** (0.1e1 / 0.3e1)
t23 = 9 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t3 ** 2
t27 = t24 * t25 * p_a_cam_omega
t29 = t1 / t22
t30 = lax_cond(t15, t16, t18)
t34 = t27 * t29 / t30 / 0.18e2
t36 = 0.135e1 < t34
t37 = lax_cond(t36, t34, 0.135e1)
t38 = t37 ** 2
t41 = t38 ** 2
t44 = t41 * t38
t47 = t41 ** 2
t59 = t47 ** 2
t63 = lax_cond(t36, 0.135e1, t34)
t64 = math.sqrt(math.pi)
t67 = math.erf(0.1e1 / t63 / 0.2e1)
t69 = t63 ** 2
t72 = math.exp(-0.1e1 / t69 / 0.4e1)
t83 = lax_cond(0.135e1 <= t34, 0.1e1 / t38 / 0.36e2 - 0.1e1 / t41 / 0.960e3 + 0.1e1 / t44 / 0.26880e5 - 0.1e1 / t47 / 0.829440e6 + 0.1e1 / t47 / t38 / 0.28385280e8 - 0.1e1 / t47 / t41 / 0.1073479680e10 + 0.1e1 / t47 / t44 / 0.44590694400e11 - 0.1e1 / t59 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t63 * (t64 * t67 + 0.2e1 * t63 * (t72 - 0.3e1 / 0.2e1 - 0.2e1 * t69 * (t72 - 0.1e1))))
t87 = 0.1e1 - t13
t88 = t87 <= p_a_zeta_threshold
t89 = t87 ** (0.1e1 / 0.3e1)
t91 = lax_cond(t88, t17, t89 * t87)
t93 = lax_cond(t88, t16, t89)
t97 = t27 * t29 / t93 / 0.18e2
t99 = 0.135e1 < t97
t100 = lax_cond(t99, t97, 0.135e1)
t101 = t100 ** 2
t104 = t101 ** 2
t107 = t104 * t101
t110 = t104 ** 2
t122 = t110 ** 2
t126 = lax_cond(t99, 0.135e1, t97)
t129 = math.erf(0.1e1 / t126 / 0.2e1)
t131 = t126 ** 2
t134 = math.exp(-0.1e1 / t131 / 0.4e1)
t145 = lax_cond(0.135e1 <= t97, 0.1e1 / t101 / 0.36e2 - 0.1e1 / t104 / 0.960e3 + 0.1e1 / t107 / 0.26880e5 - 0.1e1 / t110 / 0.829440e6 + 0.1e1 / t110 / t101 / 0.28385280e8 - 0.1e1 / t110 / t104 / 0.1073479680e10 + 0.1e1 / t110 / t107 / 0.44590694400e11 - 0.1e1 / t122 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t126 * (t64 * t129 + 0.2e1 * t126 * (t134 - 0.3e1 / 0.2e1 - 0.2e1 * t131 * (t134 - 0.1e1))))
res = -0.3e1 / 0.32e2 * t7 * t9 * t91 * t22 * t145 - 0.3e1 / 0.32e2 * t7 * t9 * t20 * t22 * t83
