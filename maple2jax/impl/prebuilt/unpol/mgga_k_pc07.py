t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = jnp.where(t8, -t9, 0)
t12 = jnp.where(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = jnp.where(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = 6 ** (0.1e1 / 0.3e1)
t26 = math.pi ** 2
t27 = t26 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = t25 / t28
t31 = 2 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = r0 ** 2
t38 = t30 * s0 * t32 / t23 / t34
t39 = 0.5e1 / 0.72e2 * t38
t47 = t25 ** 2
t50 = t47 / t27 / t26
t51 = l0 ** 2
t58 = t50 * t51 * t31 / t22 / t34 / r0 / 0.2916e4
t60 = t34 ** 2
t66 = t50 * s0 * t31 / t22 / t60 * l0 / 0.2592e4
t67 = s0 ** 2
t74 = t50 * t67 * t31 / t22 / t60 / r0 / 0.8748e4
t77 = (t58 - t66 + t74) ** 2
t79 = (0.1e1 + t39) ** 2
t83 = math.sqrt(0.1e1 + t77 / t79)
t86 = (0.1e1 + 0.5e1 / 0.648e3 * t38 + 0.5e1 / 0.54e2 * t30 * l0 * t32 / t23 / r0 + t58 - t66 + t74) / t83 - t39
t87 = params_a_a / 0.40e2
t89 = 0.39e2 / 0.40e2 * params_a_a
t93 = jnp.where(t86 < t87, t87, t86)
t95 = jnp.where(t93 < t89, t93, t89)
t96 = 0.1e1 / t95
t98 = math.exp(-params_a_a * params_a_b * t96)
t102 = math.exp(-params_a_a / (params_a_a - t95))
t104 = (0.1e1 + t102) ** params_a_b
t107 = math.exp(-params_a_a * t96)
t109 = (t107 + t102) ** params_a_b
t112 = jnp.where(t89 <= t86, 1, t98 * t104 / t109)
t113 = jnp.where(t86 <= t87, 0, t112)
t119 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (t86 * t113 + t39))
res = 0.2e1 * t119
