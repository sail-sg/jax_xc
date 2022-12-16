t1 = 0.1e1 <= p_a_zeta_threshold
t4 = jnp.logical_or(t1, r0 / 0.2e1 <= p_a_dens_threshold)
t5 = p_a_zeta_threshold - 0.1e1
t6 = -t5
t7 = jnp.where(t1, t6, 0)
t8 = jnp.where(t1, t5, t7)
t9 = t8 ** 2
t12 = 0.1e1 + t8
t16 = 3 ** (0.1e1 / 0.3e1)
t17 = t16 ** 2
t19 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t22 = 4 ** (0.1e1 / 0.3e1)
t23 = t17 / t19 * t22
t24 = 2 ** (0.1e1 / 0.3e1)
t25 = t12 <= p_a_zeta_threshold
t26 = 0.1e1 - t8
t27 = t26 <= p_a_zeta_threshold
t28 = jnp.where(t27, t6, t8)
t29 = jnp.where(t25, t5, t28)
t32 = ((0.1e1 + t29) * r0) ** (0.1e1 / 0.3e1)
t35 = t24 ** 2
t37 = r0 ** 2
t38 = r0 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t42 = math.sqrt(s0)
t43 = t42 * t24
t45 = 0.1e1 / t38 / r0
t47 = math.asinh(t43 * t45)
t58 = 0.1e1 / (0.1e1 + 0.93333333333333333332e-3 * t23 * s0 * t35 / t39 / t37 / (0.1e1 + 0.2520e-1 * t43 * t45 * t47))
t62 = jnp.where(t12 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t32 * t58 / 0.9e1)
t67 = jnp.where(t25, t6, -t8)
t68 = jnp.where(t27, t5, t67)
t71 = ((0.1e1 + t68) * r0) ** (0.1e1 / 0.3e1)
t77 = jnp.where(t26 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t71 * t58 / 0.9e1)
t78 = t62 + t77
t80 = jnp.where(t78 == 0.0e0, DBL_EPSILON, t78)
t84 = t80 ** 2
t85 = t84 ** 2
res = jnp.where(t4, 0, -0.25000000000000000000e0 * (0.1e1 - t9) * r0 * (0.360115380e1 / t80 + 0.5764e0) / (0.313901240307210000e2 / t85 + 0.149643497914092000e2 / t84 / t80 + 0.1783335908700e1 / t84))
