t1 = 3 ** (0.1e1 / 0.3e1)
t2 = 0.1e1 / math.pi
t3 = t2 ** (0.1e1 / 0.3e1)
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 + r1
t8 = t7 ** (0.1e1 / 0.3e1)
t11 = t1 * t3 * t6 / t8
t14 = math.sqrt(t11)
t17 = t11 ** 0.15e1
t19 = t1 ** 2
t20 = t3 ** 2
t22 = t8 ** 2
t25 = t19 * t20 * t5 / t22
t31 = math.log(0.1e1 + 0.16081824322151104822e2 / (0.37978500000000000000e1 * t14 + 0.89690000000000000000e0 * t11 + 0.20477500000000000000e0 * t17 + 0.12323500000000000000e0 * t25))
t33 = 0.62182e-1 * (0.1e1 + 0.53425000000000000000e-1 * t11) * t31
t34 = r0 - r1
t35 = t34 ** 2
t36 = t35 ** 2
t37 = t7 ** 2
t38 = t37 ** 2
t42 = t34 / t7
t43 = 0.1e1 + t42
t44 = t43 <= p_a_zeta_threshold
t45 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t46 = t45 * p_a_zeta_threshold
t47 = t43 ** (0.1e1 / 0.3e1)
t49 = lax_cond(t44, t46, t47 * t43)
t50 = 0.1e1 - t42
t51 = t50 <= p_a_zeta_threshold
t52 = t50 ** (0.1e1 / 0.3e1)
t54 = lax_cond(t51, t46, t52 * t50)
t56 = 2 ** (0.1e1 / 0.3e1)
t60 = (t49 + t54 - 0.2e1) / (0.2e1 * t56 - 0.2e1)
t71 = math.log(0.1e1 + 0.32164683177870697974e2 / (0.70594500000000000000e1 * t14 + 0.15494250000000000000e1 * t11 + 0.42077500000000000000e0 * t17 + 0.15629250000000000000e0 * t25))
t84 = math.log(0.1e1 + 0.29608574643216675549e2 / (0.51785000000000000000e1 * t14 + 0.90577500000000000000e0 * t11 + 0.11003250000000000000e0 * t17 + 0.12417750000000000000e0 * t25))
t85 = (0.1e1 + 0.27812500000000000000e-1 * t11) * t84
t89 = t36 / t38 * t60 * (-0.31090e-1 * (0.1e1 + 0.51370000000000000000e-1 * t11) * t71 + t33 - 0.19751789702565206229e-1 * t85)
t91 = 0.19751789702565206229e-1 * t60 * t85
t92 = math.pi ** 2
t95 = t92 ** (0.1e1 / 0.3e1)
t96 = t95 ** 2
t97 = t45 ** 2
t98 = t47 ** 2
t99 = lax_cond(t44, t97, t98)
t100 = t52 ** 2
t101 = lax_cond(t51, t97, t100)
t103 = t99 / 0.2e1 + t101 / 0.2e1
t104 = t103 ** 2
t105 = t104 * t103
t108 = 0.1e1 / t95
t110 = s0 + 0.2e1 * s1 + s2
t112 = 0.1e1 / t8 / t37
t115 = 0.1e1 / t104
t117 = 0.1e1 / t3
t118 = t117 * t5
t127 = 0.1e1 / t96
t131 = math.exp(-0.13067859477648036197e2 * (-t33 + t89 + t91) / t105 * t92 * t1 * t127)
t132 = t131 - 0.1e1
t133 = 0.1e1 / t132
t134 = t110 ** 2
t139 = t56 ** 2
t141 = t104 ** 2
t146 = 0.1e1 / t22 / t38 * t139 / t141 / t20 * t6
t155 = t112 * t56
t162 = t132 ** 2
t175 = math.log(0.1e1 + 0.88547815820543093274e0 * math.pi * t19 * t108 * (t110 * t112 * t56 * t115 * t19 * t118 / 0.96e2 + 0.86472476387249114526e-3 * math.pi * t108 * t133 * t134 * t146) / (0.1e1 + 0.27671192443919716648e-1 * math.pi * t1 * t108 * t133 * t110 * t155 * t115 * t117 * t5 + 0.76569489126843962094e-3 * t92 * t19 * t127 / t162 * t134 * t146))
t193 = 9 ** (0.1e1 / 0.3e1)
t194 = t193 ** 2
t204 = math.exp(-0.25e2 / 0.18e2 * t2 * t5 * t194 * t3 / t22 / t37 * t104 * t110 * t56)
res = -t33 + t89 + t91 + 0.25507875555555555556e-1 / t92 * t19 * t96 * t105 * t175 + t2 * t95 * ((0.2568e1 + 0.58165000000000000000e1 * t11 + 0.18472500000000000000e-2 * t25) / (0.1000e4 + 0.21807500000000000000e4 * t11 + 0.11800000000000000000e3 * t25) - 0.18535714285714285714e-2) * t103 * t110 * t155 * t118 * t204 / 0.2e1
