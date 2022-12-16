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
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t33 = 0.1e1 / t32
t34 = t29 * t33
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t39 = 0.1e1 / t37 / t35
t41 = t34 * s0 * t39
t45 = 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t41)
t47 = t29 ** 2
t50 = t47 / t31 / t30
t51 = s0 ** 2
t52 = t35 ** 2
t58 = t50 * t51 / t36 / t52 / r0 / 0.576e3
t60 = t30 ** 2
t61 = 0.1e1 / t60
t64 = t52 ** 2
t83 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.18040e1 - t45 + (t41 / 0.24e2 + t58) / (0.1e1 + t58 + t61 * t51 * s0 / t64 / 0.2304e4) * (-(0.18040e1 - t45) * t29 * t33 * s0 * t39 / 0.24e2 + 0.6525e-1)))
t85 = jnp.where(t10, t15, -t17)
t86 = jnp.where(t14, t11, t85)
t87 = 0.1e1 + t86
t89 = t87 ** (0.1e1 / 0.3e1)
t91 = jnp.where(t87 <= p_a_zeta_threshold, t23, t89 * t87)
t93 = r1 ** 2
t94 = r1 ** (0.1e1 / 0.3e1)
t95 = t94 ** 2
t97 = 0.1e1 / t95 / t93
t99 = t34 * s2 * t97
t103 = 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t99)
t105 = s2 ** 2
t106 = t93 ** 2
t112 = t50 * t105 / t94 / t106 / r1 / 0.576e3
t116 = t106 ** 2
t135 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t91 * t27 * (0.18040e1 - t103 + (t99 / 0.24e2 + t112) / (0.1e1 + t112 + t61 * t105 * s2 / t116 / 0.2304e4) * (-(0.18040e1 - t103) * t29 * t33 * s2 * t97 / 0.24e2 + 0.6525e-1)))
res = t83 + t135
