t1 = math.log(0.2e1)
t2 = 0.1e1 - t1
t3 = math.pi ** 2
t4 = 0.1e1 / t3
t5 = t2 * t4
t6 = t1 / 0.6e1
t7 = 0.90154267736969571405e0 * t4
t9 = 0.1e1 / t2
t13 = math.exp(-0.2e1 * (-0.71100e-1 + t6 - t7) * t9 * t3)
t14 = 3 ** (0.1e1 / 0.3e1)
t15 = t14 ** 2
t17 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t18 = t17 ** 2
t20 = 4 ** (0.1e1 / 0.3e1)
t21 = r0 + r1
t22 = t21 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t28 = math.exp(-t15 * t18 * t20 / t23 / 0.40000e5)
t29 = 0.1e1 - t28
t30 = math.pi ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = 9 ** (0.1e1 / 0.3e1)
t34 = 0.1e1 / t31 * t33
t35 = t20 ** 2
t41 = t13 / 0.2e1
t47 = 0.1e1 / t17
t49 = t47 * t20 * t22
t53 = math.sqrt(0.4e1)
t55 = t14 * t17
t56 = 0.1e1 / t22
t58 = t55 * t35 * t56
t59 = math.sqrt(t58)
t63 = t29 * t9 * t3 * t53 / t59 / t58
t65 = t33 ** 2
t66 = t65 * t20
t67 = t31 * t3
t78 = 0.1e1 / t18 * t35 * t23
t82 = math.log(0.1e1 + (t13 - 0.2e1 * t29 * ((-0.9e0 + 0.3e1 / 0.16e2 * t34 * t35) * t9 * t3 + t41)) * t15 * t49 / 0.3e1 - 0.120e2 * t63 + (t13 - 0.2e1 * t29 * (-0.3e1 / 0.40e2 * t66 * t67 * t9 + t41)) * t14 * t78 / 0.3e1)
t84 = t5 * t82 / 0.2e1
t86 = t56 * t28
t87 = 4 ** (0.1e1 / 0.4e1)
t88 = t87 ** 2
t90 = t58 ** (0.1e1 / 0.4e1)
t95 = 0.1e1 / (t28 + 0.5e1 / 0.8e1 * t88 * t87 * t90 * t58)
t98 = 0.1e1 / t30 / t3 / math.pi
t100 = 0.12e2 * t1
t108 = math.log(0.1e1 + t15 * t47 * t20 * t22 / 0.3e1)
t116 = t55 * t35 * t86 * t95 * (-t66 * t98 * (0.7e1 / 0.6e1 * t3 - t100 - 0.1e1) * t108 / 0.36e2 - 0.1e-1) / 0.4e1
t121 = math.exp(-0.4e1 * (-0.49917e-1 + t6 - t7) * t9 * t3)
t122 = 2 ** (0.1e1 / 0.3e1)
t130 = t121 / 0.2e1
t139 = t122 ** 2
t152 = math.log(0.1e1 + (t121 - 0.2e1 * t29 * (0.2e1 * (-0.9e0 + 0.3e1 / 0.16e2 * t34 * t35 * t122) * t9 * t3 + t130)) * t15 * t49 / 0.3e1 - 0.240e2 * t63 + (t121 - 0.2e1 * t29 * (-0.3e1 / 0.20e2 * t66 * t67 * t139 * t9 + t130)) * t14 * t78 / 0.3e1)
t168 = (r0 - r1) / t21
t169 = 0.1e1 + t168
t171 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t172 = t171 * p_a_zeta_threshold
t173 = t169 ** (0.1e1 / 0.3e1)
t175 = lax_cond(t169 <= p_a_zeta_threshold, t172, t173 * t169)
t176 = 0.1e1 - t168
t178 = t176 ** (0.1e1 / 0.3e1)
t180 = lax_cond(t176 <= p_a_zeta_threshold, t172, t178 * t176)
res = -t84 + t116 + (-t5 * t152 / 0.4e1 - t55 * t86 * t95 * t139 * t65 * t98 * (0.13e2 / 0.12e2 * t3 - t100 + 0.1e1 / 0.2e1) * t108 / 0.144e3 + t84 - t116) * (t175 + t180 - 0.2e1) / (0.2e1 * t122 - 0.2e1)
