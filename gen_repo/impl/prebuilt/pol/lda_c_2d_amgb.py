t1 = math.sqrt(math.pi)
t2 = 0.1e1 / t1
t3 = r0 + r1
t4 = math.sqrt(t3)
t6 = t2 / t4
t9 = 0.1e1 / t3
t10 = 0.1e1 / math.pi * t9
t16 = 0.1e1 / t1 / math.pi / t4 / t3
t20 = t6 ** 0.15e1
t27 = math.log(0.1e1 + 0.1e1 / (0.10022e1 * t6 - 0.2069e-1 * t20 + 0.33997e0 * t10 + 0.1747e-1 * t16))
t39 = math.log(0.1e1 + 0.1e1 / (0.4133e0 * t6 + 0.668467e-1 * t10 + 0.7799e-3 * t16))
t42 = r0 - r1
t43 = t42 ** 2
t45 = t3 ** 2
t46 = 0.1e1 / t45
t57 = math.log(0.1e1 + 0.1e1 / (0.1424301e1 * t6 + 0.1163099e1 * t16))
t60 = t43 ** 2
t62 = t45 ** 2
t63 = 0.1e1 / t62
t66 = math.exp(-0.13386e1 * t6)
t68 = math.sqrt(0.2e1)
t71 = t42 * t9
t72 = 0.1e1 + t71
t74 = math.sqrt(p_a_zeta_threshold)
t75 = t74 * p_a_zeta_threshold
t76 = math.sqrt(t72)
t78 = lax_cond(t72 <= p_a_zeta_threshold, t75, t76 * t72)
t80 = 0.1e1 - t71
t82 = math.sqrt(t80)
t84 = lax_cond(t80 <= p_a_zeta_threshold, t75, t82 * t80)
res = -0.1925e0 + (0.863136e-1 * t6 + 0.572384e-1 * t10 + 0.3362975e-2 * t16) * t27 + (0.117331e0 + (-0.3394e-1 * t6 - 0.766765e-2 * t10 - 0.915064469e-4 * t16) * t39) * t43 * t46 + (0.234188e-1 + (-0.37093e-1 * t6 + 0.163618e-1 * t10 - 0.272383828612e-1 * t16) * t57) * t60 * t63 - 0.4e1 / 0.3e1 * (t66 - 0.1e1) * t68 * t2 * t4 * (t78 / 0.2e1 + t84 / 0.2e1 - 0.1e1 - 0.3e1 / 0.8e1 * t43 * t46 - 0.3e1 / 0.128e3 * t60 * t63)
