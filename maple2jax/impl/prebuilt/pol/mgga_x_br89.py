t2 = r0 + r1
t3 = 0.1e1 / t2
t6 = 0.2e1 * r0 * t3 <= p_a_zeta_threshold
t7 = p_a_zeta_threshold - 0.1e1
t10 = 0.2e1 * r1 * t3 <= p_a_zeta_threshold
t11 = -t7
t13 = (r0 - r1) * t3
t14 = jnp.where(t10, t11, t13)
t15 = jnp.where(t6, t7, t14)
t16 = 0.1e1 + t15
t18 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t19 = t18 * p_a_zeta_threshold
t20 = t16 ** (0.1e1 / 0.3e1)
t22 = jnp.where(t16 <= p_a_zeta_threshold, t19, t20 * t16)
t23 = t2 ** (0.1e1 / 0.3e1)
t26 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t28 = 4 ** (0.1e1 / 0.3e1)
t29 = 0.1e1 / t26 * t28
t31 = r0 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = 0.1e1 / t32 / r0
t41 = r0 ** 2
t46 = l0 * t34 / 0.6e1 - 0.2e1 / 0.3e1 * params_a_gamma * tau0 * t34 + params_a_gamma * s0 / t32 / t41 / 0.12e2
t47 = abs(t46)
t50 = jnp.where(0.0e0 < t46, 0.50e-12, -0.50e-12)
t51 = jnp.where(t47 < 0.50e-12, t50, t46)
t52 = br89_x(t51)
t54 = math.exp(t52 / 0.3e1)
t55 = math.exp(-t52)
t62 = 6 ** (0.1e1 / 0.3e1)
t63 = t62 ** 2
t64 = math.pi ** 2
t65 = t64 ** (0.1e1 / 0.3e1)
t66 = t65 ** 2
t68 = 0.3e1 / 0.10e2 * t63 * t66
t69 = tau0 * t34
t70 = t68 - t69
t71 = t68 + t69
t74 = t70 ** 2
t76 = t71 ** 2
t81 = t74 ** 2
t83 = t76 ** 2
t94 = jnp.where(r0 <= p_a_dens_threshold, 0, -t22 * t23 * t29 * t54 * (0.1e1 - t55 * (0.1e1 + t52 / 0.2e1)) / t52 * (0.1e1 + params_a_at * (t70 / t71 - 0.2e1 * t74 * t70 / t76 / t71 + t81 * t70 / t83 / t71)) / 0.4e1)
t96 = jnp.where(t6, t11, -t13)
t97 = jnp.where(t10, t7, t96)
t98 = 0.1e1 + t97
t100 = t98 ** (0.1e1 / 0.3e1)
t102 = jnp.where(t98 <= p_a_zeta_threshold, t19, t100 * t98)
t105 = r1 ** (0.1e1 / 0.3e1)
t106 = t105 ** 2
t108 = 0.1e1 / t106 / r1
t115 = r1 ** 2
t120 = l1 * t108 / 0.6e1 - 0.2e1 / 0.3e1 * params_a_gamma * tau1 * t108 + params_a_gamma * s2 / t106 / t115 / 0.12e2
t121 = abs(t120)
t124 = jnp.where(0.0e0 < t120, 0.50e-12, -0.50e-12)
t125 = jnp.where(t121 < 0.50e-12, t124, t120)
t126 = br89_x(t125)
t128 = math.exp(t126 / 0.3e1)
t129 = math.exp(-t126)
t136 = tau1 * t108
t137 = t68 - t136
t138 = t68 + t136
t141 = t137 ** 2
t143 = t138 ** 2
t148 = t141 ** 2
t150 = t143 ** 2
t161 = jnp.where(r1 <= p_a_dens_threshold, 0, -t102 * t23 * t29 * t128 * (0.1e1 - t129 * (0.1e1 + t126 / 0.2e1)) / t126 * (0.1e1 + params_a_at * (t137 / t138 - 0.2e1 * t141 * t137 / t143 / t138 + t148 * t137 / t150 / t138)) / 0.4e1)
res = t94 + t161
