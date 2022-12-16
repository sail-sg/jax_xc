t1 = 3 ** (0.1e1 / 0.3e1)
t2 = t1 ** 2
t5 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t7 = 4 ** (0.1e1 / 0.3e1)
t9 = r0 ** (0.1e1 / 0.3e1)
t10 = 0.1e1 / t5 * t7 * t9
t14 = t5 ** 2
t16 = t7 ** 2
t18 = t9 ** 2
t19 = 0.1e1 / t14 * t16 * t18
t23 = math.log(0.1e1 + params_a_bp * t2 * t10 / 0.3e1 + params_a_cp * t1 * t19 / 0.3e1)
t24 = params_a_ap * t23
t32 = math.log(0.1e1 + params_a_bf * t2 * t10 / 0.3e1 + params_a_cf * t1 * t19 / 0.3e1)
t36 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t38 = jnp.where(0.1e1 <= p_a_zeta_threshold, t37, 1)
t39 = t38 ** 2
res = t24 + (params_a_af * t32 - t24) * (-0.2e1 * t39 * t38 + 0.2e1)
