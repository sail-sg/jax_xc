t1 = 3 ** (0.1e1 / 0.3e1)
t2 = 0.1e1 / math.pi
t3 = t2 ** (0.1e1 / 0.3e1)
t4 = t1 * t3
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 ** (0.1e1 / 0.3e1)
t8 = 0.1e1 / t7
t9 = t6 * t8
t10 = t4 * t9
t11 = t10 / 0.4e1
t12 = 0.1e1 <= t11
t13 = math.sqrt(t10)
t19 = math.log(t11)
t22 = t4 * t9 * t19
t26 = lax_cond(t12, -0.1423e0 / (0.1e1 + 0.52645000000000000000e0 * t13 + 0.83350000000000000000e-1 * t10), 0.311e-1 * t19 - 0.48e-1 + 0.50000000000000000000e-3 * t22 - 0.29000000000000000000e-2 * t10)
t36 = lax_cond(t12, -0.843e-1 / (0.1e1 + 0.69905000000000000000e0 * t13 + 0.65275000000000000000e-1 * t10), 0.1555e-1 * t19 - 0.269e-1 + 0.17500000000000000000e-3 * t22 - 0.12000000000000000000e-2 * t10)
t38 = 0.1e1 <= p_a_zeta_threshold
t39 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t41 = lax_cond(t38, t39 * p_a_zeta_threshold, 1)
t45 = 2 ** (0.1e1 / 0.3e1)
t50 = r0 ** 2
t58 = t3 * t6 * t8
t61 = t1 ** 2
t63 = t3 ** 2
t65 = t7 ** 2
t67 = t63 * t5 / t65
t84 = params_a_aa + (params_a_bb + params_a_malpha * t1 * t58 / 0.4e1 + params_a_mbeta * t61 * t67 / 0.4e1) / (0.1e1 + params_a_mgamma * t1 * t58 / 0.4e1 + params_a_mdelta * t61 * t67 / 0.4e1 + 0.75000000000000000000e4 * params_a_mbeta * t2 / r0)
t86 = math.sqrt(s0)
t88 = r0 ** (0.1e1 / 0.6e1)
t93 = math.exp(-params_a_ftilde * (params_a_aa + params_a_bb) / t84 * t86 / t88 / r0)
t95 = t39 ** 2
t97 = lax_cond(t38, t95 * p_a_zeta_threshold, 1)
t98 = math.sqrt(t97)
res = t26 + (t36 - t26) * (0.2e1 * t41 - 0.2e1) / (0.2e1 * t45 - 0.2e1) + s0 / t7 / t50 * t93 * t84 / t98
