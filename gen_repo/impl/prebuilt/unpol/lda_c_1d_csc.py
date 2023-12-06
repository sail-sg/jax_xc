t1 = 0.1e1 / r0
t2 = t1 / 0.2e1
t4 = r0 ** 2
t14 = t2 ** params_a_para[9]
t17 = math.log(0.1e1 + params_a_para[7] * t1 / 0.2e1 + params_a_para[8] * t14)
t25 = t2 ** params_a_para[5]
t30 = t2 ** params_a_para[6]
res = -(t2 + params_a_para[4] / t4 / 0.4e1) * t17 / (params_a_para[1] * t1 + 0.2e1 * params_a_para[2] * t25 + 0.2e1 * params_a_para[3] * t30 + 0.2e1 * params_a_para[0])
