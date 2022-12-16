t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t22 = 2 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t24 = r0 ** 2
t25 = t20 ** 2
t28 = t23 / t25 / t24
t30 = s0 ** 2
t32 = t24 ** 2
t49 = params_a_A + params_a_B * (0.1e1 - 0.1e1 / (0.1e1 + params_a_C * s0 * t28 + 0.2e1 * params_a_D * t30 * t22 / t20 / t32 / r0)) * (0.1e1 - 0.1e1 / (params_a_E * s0 * t28 + 0.1e1))
t51 = t3 ** 2
t54 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t56 = 4 ** (0.1e1 / 0.3e1)
t61 = math.sqrt(math.pi * t51 / t54 * t56 / t49)
t65 = (t12 * r0) ** (0.1e1 / 0.3e1)
t69 = p_a_cam_omega / t61 * t22 / t65 / 0.2e1
t71 = 0.135e1 < t69
t72 = jnp.where(t71, t69, 0.135e1)
t73 = t72 ** 2
t76 = t73 ** 2
t79 = t76 * t73
t82 = t76 ** 2
t94 = t82 ** 2
t98 = jnp.where(t71, 0.135e1, t69)
t99 = math.sqrt(math.pi)
t102 = math.erf(0.1e1 / t98 / 0.2e1)
t104 = t98 ** 2
t107 = math.exp(-0.1e1 / t104 / 0.4e1)
t118 = jnp.where(0.135e1 <= t69, 0.1e1 / t73 / 0.36e2 - 0.1e1 / t76 / 0.960e3 + 0.1e1 / t79 / 0.26880e5 - 0.1e1 / t82 / 0.829440e6 + 0.1e1 / t82 / t73 / 0.28385280e8 - 0.1e1 / t82 / t76 / 0.1073479680e10 + 0.1e1 / t82 / t79 / 0.44590694400e11 - 0.1e1 / t94 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t98 * (t99 * t102 + 0.2e1 * t98 * (t107 - 0.3e1 / 0.2e1 - 0.2e1 * t104 * (t107 - 0.1e1))))
t124 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * t49 * (-p_a_cam_beta * t118 - p_a_cam_alpha + 0.1e1))
res = 0.2e1 * t124
