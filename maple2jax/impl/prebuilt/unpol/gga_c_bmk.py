t3 = 0.1e1 <= p_a_zeta_threshold
t4 = jnp.logical_or(r0 / 0.2e1 <= p_a_dens_threshold, t3)
t5 = lax_cond(t3, p_a_zeta_threshold, 1)
t6 = 3 ** (0.1e1 / 0.3e1)
t8 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t9 = t6 * t8
t10 = 4 ** (0.1e1 / 0.3e1)
t11 = t10 ** 2
t13 = r0 ** (0.1e1 / 0.3e1)
t14 = 0.1e1 / t13
t15 = 2 ** (0.1e1 / 0.3e1)
t17 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t19 = lax_cond(t3, 0.1e1 / t17, 1)
t21 = t9 * t11 * t14 * t15 * t19
t24 = math.sqrt(t21)
t27 = t21 ** 0.15e1
t29 = t6 ** 2
t30 = t8 ** 2
t31 = t29 * t30
t33 = t13 ** 2
t34 = 0.1e1 / t33
t35 = t15 ** 2
t37 = t19 ** 2
t39 = t31 * t10 * t34 * t35 * t37
t45 = math.log(0.1e1 + 0.16081824322151104822e2 / (0.37978500000000000000e1 * t24 + 0.89690000000000000000e0 * t21 + 0.20477500000000000000e0 * t27 + 0.12323500000000000000e0 * t39))
t47 = 0.62182e-1 * (0.1e1 + 0.53425000000000000000e-1 * t21) * t45
t49 = t17 * p_a_zeta_threshold
t51 = lax_cond(0.2e1 <= p_a_zeta_threshold, t49, 0.2e1 * t15)
t53 = lax_cond(0.0e0 <= p_a_zeta_threshold, t49, 0)
t57 = 0.1e1 / (0.2e1 * t15 - 0.2e1)
t58 = (t51 + t53 - 0.2e1) * t57
t69 = math.log(0.1e1 + 0.32164683177870697974e2 / (0.70594500000000000000e1 * t24 + 0.15494250000000000000e1 * t21 + 0.42077500000000000000e0 * t27 + 0.15629250000000000000e0 * t39))
t82 = math.log(0.1e1 + 0.29608574643216675549e2 / (0.51785000000000000000e1 * t24 + 0.90577500000000000000e0 * t21 + 0.11003250000000000000e0 * t27 + 0.12417750000000000000e0 * t39))
t83 = (0.1e1 + 0.27812500000000000000e-1 * t21) * t82
t92 = lax_cond(t4, 0, t5 * (-t47 + t58 * (-0.31090e-1 * (0.1e1 + 0.51370000000000000000e-1 * t21) * t69 + t47 - 0.19751789702565206229e-1 * t83) + 0.19751789702565206229e-1 * t58 * t83) / 0.2e1)
t96 = r0 ** 2
t98 = 0.1e1 / t33 / t96
t99 = t35 * t98
t101 = s0 * t35 * t98
t103 = 0.1e1 + 0.2e0 * t101
t109 = s0 ** 2
t111 = t96 ** 2
t115 = t15 / t13 / t111 / r0
t116 = t103 ** 2
t122 = t109 * s0
t124 = t111 ** 2
t125 = 0.1e1 / t124
t132 = t109 ** 2
t137 = t35 / t33 / t124 / t96
t138 = t116 ** 2
t147 = t9 * t11 * t14
t150 = math.sqrt(t147)
t153 = t147 ** 0.15e1
t156 = t31 * t10 * t34
t162 = math.log(0.1e1 + 0.16081824322151104822e2 / (0.37978500000000000000e1 * t150 + 0.89690000000000000000e0 * t147 + 0.20477500000000000000e0 * t153 + 0.12323500000000000000e0 * t156))
t165 = lax_cond(t3, t49, 1)
t179 = math.log(0.1e1 + 0.29608574643216675549e2 / (0.51785000000000000000e1 * t150 + 0.90577500000000000000e0 * t147 + 0.11003250000000000000e0 * t153 + 0.12417750000000000000e0 * t156))
t189 = 0.1e1 + 0.6e-2 * t101
t196 = t189 ** 2
t210 = t196 ** 2
res = 0.2e1 * t92 * (params_a_c_ss[0] + 0.2e0 * params_a_c_ss[1] * s0 * t99 / t103 + 0.8e-1 * params_a_c_ss[2] * t109 * t115 / t116 + 0.32e-1 * params_a_c_ss[3] * t122 * t125 / t116 / t103 + 0.64e-2 * params_a_c_ss[4] * t132 * t137 / t138) + (-0.62182e-1 * (0.1e1 + 0.53425000000000000000e-1 * t147) * t162 + 0.19751789702565206229e-1 * (0.2e1 * t165 - 0.2e1) * t57 * (0.1e1 + 0.27812500000000000000e-1 * t147) * t179 - 0.2e1 * t92) * (params_a_c_ab[0] + 0.6e-2 * params_a_c_ab[1] * s0 * t99 / t189 + 0.72e-4 * params_a_c_ab[2] * t109 * t115 / t196 + 0.864e-6 * params_a_c_ab[3] * t122 * t125 / t196 / t189 + 0.5184e-8 * params_a_c_ab[4] * t132 * t137 / t210)
