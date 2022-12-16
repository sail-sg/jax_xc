t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = jnp.where(t15, t16, t18)
t20 = jnp.where(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = jnp.where(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = 6 ** (0.1e1 / 0.3e1)
t35 = math.pi ** 2
t36 = t35 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t38 = 0.1e1 / t37
t39 = params_a_c1 * t33 * t38
t40 = r0 ** 2
t41 = r0 ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t44 = 0.1e1 / t42 / t40
t46 = params_a_a * t33
t51 = 0.1e1 + t46 * t38 * s0 * t44 / 0.24e2
t56 = t33 ** 2
t60 = params_a_c2 * t56 / t36 / t35
t61 = s0 ** 2
t62 = t40 ** 2
t67 = t51 ** 2
t72 = t35 ** 2
t74 = params_a_c3 / t72
t76 = t62 ** 2
t88 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + t39 * s0 * t44 / t51 / 0.24e2 + t60 * t61 / t41 / t62 / r0 / t67 / 0.576e3 + t74 * t61 * s0 / t76 / t67 / t51 / 0.2304e4))
t90 = jnp.where(t11, t16, -t18)
t91 = jnp.where(t15, t12, t90)
t92 = 0.1e1 + t91
t94 = t92 ** (0.1e1 / 0.3e1)
t95 = t94 ** 2
t97 = jnp.where(t92 <= p_a_zeta_threshold, t25, t95 * t92)
t99 = r1 ** 2
t100 = r1 ** (0.1e1 / 0.3e1)
t101 = t100 ** 2
t103 = 0.1e1 / t101 / t99
t109 = 0.1e1 + t46 * t38 * s2 * t103 / 0.24e2
t114 = s2 ** 2
t115 = t99 ** 2
t120 = t109 ** 2
t126 = t115 ** 2
t138 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t97 * t31 * (0.1e1 + t39 * s2 * t103 / t109 / 0.24e2 + t60 * t114 / t100 / t115 / r1 / t120 / 0.576e3 + t74 * t114 * s2 / t126 / t120 / t109 / 0.2304e4))
res = t88 + t138
