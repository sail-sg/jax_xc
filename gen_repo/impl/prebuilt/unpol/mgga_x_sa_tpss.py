t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = math.sqrt(0.5e1)
t22 = math.pi * t21
t23 = 2 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t26 = t19 ** 2
t31 = r0 ** 2
t34 = s0 * t24 / t26 / t31
t37 = 6 ** (0.1e1 / 0.3e1)
t38 = (tau0 * t24 / t26 / r0 - t34 / 0.8e1) * t37
t39 = math.pi ** 2
t40 = t39 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t42 = 0.1e1 / t41
t43 = t38 * t42
t46 = math.sqrt(0.5e1 * t43 + 0.9e1)
t47 = 0.5e1 / 0.9e1 * t43
t49 = math.log(t47 + 0.348e0)
t51 = math.sqrt(0.2413e1 + t49)
t53 = t46 / t51
t56 = s0 ** 2
t58 = t56 / t31
t59 = tau0 ** 2
t60 = 0.1e1 / t59
t61 = t58 * t60
t64 = (0.1e1 + t61 / 0.64e2) ** 2
t74 = t47 - 0.1e1
t79 = math.sqrt(0.1e1 + 0.22222222222222222222e0 * t38 * t42 * t74)
t84 = t37 * t42 * t34
t86 = 0.9e1 / 0.20e2 * t74 / t79 + t84 / 0.36e2
t87 = t86 ** 2
t90 = t37 ** 2
t95 = t31 ** 2
t100 = t90 / t40 / t39 * t56 * t23 / t19 / t95 / r0
t103 = math.sqrt(0.162e3 * t61 + 0.100e3 * t100)
t114 = t39 ** 2
t118 = t95 ** 2
t125 = (0.1e1 + 0.51656585037899841583e-1 * t84) ** 2
t141 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + 0.2e1 / 0.45e2 * t22 * t53 * (0.1e1 - 0.2e1 / 0.45e2 * t22 * t53 / (0.2e1 / 0.45e2 * t22 * t53 + ((0.10e2 / 0.81e2 + 0.24858750000000000000e-1 * t58 * t60 / t64) * t37 * t42 * t34 / 0.24e2 + 0.146e3 / 0.2025e4 * t87 - 0.73e2 / 0.97200e5 * t86 * t103 + 0.25e2 / 0.104976e6 / math.pi * t21 / t46 * t51 * t100 + 0.17218861679299947194e-2 * t61 + 0.58574109375000000000e-3 / t114 * t56 * s0 / t118) / t125))))
res = 0.2e1 * t141
