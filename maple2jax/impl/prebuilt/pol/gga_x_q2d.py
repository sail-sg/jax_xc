t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = jnp.where(t14, t15, t17)
t19 = jnp.where(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = jnp.where(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t28 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t41 = t34 * s0 / t37 / t35
t47 = t29 ** 2
t50 = t47 / t31 / t30
t51 = s0 ** 2
t52 = t35 ** 2
t62 = t47 / t31
t63 = math.sqrt(s0)
t68 = (t62 * t63 / t36 / r0) ** 0.35e1
t75 = t30 ** 2
t76 = 0.1e1 / t75
t79 = t52 ** 2
t88 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t41)) * (0.100e3 - t50 * t51 / t36 / t52 / r0 / 0.576e3) + 0.87153829697982569831e-4 * t68 * (0.1e1 + t41 / 0.24e2)) / (0.100e3 + t76 * t51 * s0 / t79 / 0.2304e4))
t90 = jnp.where(t10, t15, -t17)
t91 = jnp.where(t14, t11, t90)
t92 = 0.1e1 + t91
t94 = t92 ** (0.1e1 / 0.3e1)
t96 = jnp.where(t92 <= p_a_zeta_threshold, t23, t94 * t92)
t98 = r1 ** 2
t99 = r1 ** (0.1e1 / 0.3e1)
t100 = t99 ** 2
t104 = t34 * s2 / t100 / t98
t110 = s2 ** 2
t111 = t98 ** 2
t120 = math.sqrt(s2)
t125 = (t62 * t120 / t99 / r1) ** 0.35e1
t134 = t111 ** 2
t143 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t96 * t28 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t104)) * (0.100e3 - t50 * t110 / t99 / t111 / r1 / 0.576e3) + 0.87153829697982569831e-4 * t125 * (0.1e1 + t104 / 0.24e2)) / (0.100e3 + t76 * t110 * s2 / t134 / 0.2304e4))
res = t88 + t143
