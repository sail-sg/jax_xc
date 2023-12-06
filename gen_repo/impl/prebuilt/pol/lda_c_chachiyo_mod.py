t1 = 3 ** (0.1e1 / 0.3e1)
t2 = t1 ** 2
t5 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t7 = 4 ** (0.1e1 / 0.3e1)
t9 = r0 + r1
t10 = t9 ** (0.1e1 / 0.3e1)
t11 = 0.1e1 / t5 * t7 * t10
t15 = t5 ** 2
t17 = t7 ** 2
t19 = t10 ** 2
t20 = 0.1e1 / t15 * t17 * t19
t24 = math.log(0.1e1 + params_a_bp * t2 * t11 / 0.3e1 + params_a_cp * t1 * t20 / 0.3e1)
t25 = params_a_ap * t24
t33 = math.log(0.1e1 + params_a_bf * t2 * t11 / 0.3e1 + params_a_cf * t1 * t20 / 0.3e1)
t38 = (r0 - r1) / t9
t39 = 0.1e1 + t38
t41 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t43 = t39 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t45 = lax_cond(t39 <= p_a_zeta_threshold, t42, t44)
t46 = 0.1e1 - t38
t48 = t46 ** (0.1e1 / 0.3e1)
t49 = t48 ** 2
t50 = lax_cond(t46 <= p_a_zeta_threshold, t42, t49)
t52 = t45 / 0.2e1 + t50 / 0.2e1
t53 = t52 ** 2
res = t25 + (params_a_af * t33 - t25) * (-0.2e1 * t53 * t52 + 0.2e1)
