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
t36 = 0.192e1 < t34
t37 = lax_cond(t36, t34, 0.192e1)
t38 = t37 ** 2
t41 = t38 ** 2
t44 = t41 * t38
t47 = t41 ** 2
t50 = t47 * t38
t53 = t47 * t41
t56 = t47 * t44
t59 = t47 ** 2
t83 = t59 ** 2
t92 = 0.1e1 / t38 / 0.9e1 - 0.1e1 / t41 / 0.30e2 + 0.1e1 / t44 / 0.70e2 - 0.1e1 / t47 / 0.135e3 + 0.1e1 / t50 / 0.231e3 - 0.1e1 / t53 / 0.364e3 + 0.1e1 / t56 / 0.540e3 - 0.1e1 / t59 / 0.765e3 + 0.1e1 / t59 / t38 / 0.1045e4 - 0.1e1 / t59 / t41 / 0.1386e4 + 0.1e1 / t59 / t44 / 0.1794e4 - 0.1e1 / t59 / t47 / 0.2275e4 + 0.1e1 / t59 / t50 / 0.2835e4 - 0.1e1 / t59 / t53 / 0.3480e4 + 0.1e1 / t59 / t56 / 0.4216e4 - 0.1e1 / t83 / 0.5049e4 + 0.1e1 / t83 / t38 / 0.5985e4 - 0.1e1 / t83 / t41 / 0.7030e4
t93 = lax_cond(t36, 0.192e1, t34)
t94 = math.atan2(0.1e1, t93)
t95 = t93 ** 2
t99 = math.log(0.1e1 + 0.1e1 / t95)
t108 = lax_cond(0.192e1 <= t34, t92, 0.1e1 - 0.8e1 / 0.3e1 * t93 * (t94 + t93 * (0.1e1 - (t95 + 0.3e1) * t99) / 0.4e1))
t112 = 0.1e1 - t13
t113 = t112 <= p_a_zeta_threshold
t114 = t112 ** (0.1e1 / 0.3e1)
t116 = lax_cond(t113, t17, t114 * t112)
t118 = lax_cond(t113, t16, t114)
t122 = t27 * t29 / t118 / 0.18e2
t124 = 0.192e1 < t122
t125 = lax_cond(t124, t122, 0.192e1)
t126 = t125 ** 2
t129 = t126 ** 2
t132 = t129 * t126
t135 = t129 ** 2
t138 = t135 * t126
t141 = t135 * t129
t144 = t135 * t132
t147 = t135 ** 2
t171 = t147 ** 2
t180 = 0.1e1 / t126 / 0.9e1 - 0.1e1 / t129 / 0.30e2 + 0.1e1 / t132 / 0.70e2 - 0.1e1 / t135 / 0.135e3 + 0.1e1 / t138 / 0.231e3 - 0.1e1 / t141 / 0.364e3 + 0.1e1 / t144 / 0.540e3 - 0.1e1 / t147 / 0.765e3 + 0.1e1 / t147 / t126 / 0.1045e4 - 0.1e1 / t147 / t129 / 0.1386e4 + 0.1e1 / t147 / t132 / 0.1794e4 - 0.1e1 / t147 / t135 / 0.2275e4 + 0.1e1 / t147 / t138 / 0.2835e4 - 0.1e1 / t147 / t141 / 0.3480e4 + 0.1e1 / t147 / t144 / 0.4216e4 - 0.1e1 / t171 / 0.5049e4 + 0.1e1 / t171 / t126 / 0.5985e4 - 0.1e1 / t171 / t129 / 0.7030e4
t181 = lax_cond(t124, 0.192e1, t122)
t182 = math.atan2(0.1e1, t181)
t183 = t181 ** 2
t187 = math.log(0.1e1 + 0.1e1 / t183)
t196 = lax_cond(0.192e1 <= t122, t180, 0.1e1 - 0.8e1 / 0.3e1 * t181 * (t182 + t181 * (0.1e1 - (t183 + 0.3e1) * t187) / 0.4e1))
res = -0.3e1 / 0.32e2 * t7 * t9 * t20 * t22 * t108 - 0.3e1 / 0.32e2 * t7 * t9 * t116 * t22 * t196
