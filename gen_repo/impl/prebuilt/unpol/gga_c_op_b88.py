t1 = 0.1e1 <= p_a_zeta_threshold
t3 = r0 / 0.2e1 <= p_a_dens_threshold
t4 = jnp.logical_and(t3, t3)
t5 = jnp.logical_or(t1, t4)
t6 = p_a_zeta_threshold - 0.1e1
t7 = -t6
t8 = lax_cond(t1, t7, 0)
t9 = lax_cond(t1, t6, t8)
t10 = t9 ** 2
t13 = 0.1e1 + t9
t17 = 3 ** (0.1e1 / 0.3e1)
t18 = t17 ** 2
t20 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t23 = 4 ** (0.1e1 / 0.3e1)
t24 = t18 / t20 * t23
t25 = 2 ** (0.1e1 / 0.3e1)
t26 = t13 <= p_a_zeta_threshold
t27 = 0.1e1 - t9
t28 = t27 <= p_a_zeta_threshold
t29 = lax_cond(t28, t7, t9)
t30 = lax_cond(t26, t6, t29)
t33 = ((0.1e1 + t30) * r0) ** (0.1e1 / 0.3e1)
t36 = t25 ** 2
t38 = r0 ** 2
t39 = r0 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
t43 = math.sqrt(s0)
t44 = t43 * t25
t46 = 0.1e1 / t39 / r0
t48 = math.asinh(t44 * t46)
t59 = 0.1e1 / (0.1e1 + 0.93333333333333333332e-3 * t24 * s0 * t36 / t40 / t38 / (0.1e1 + 0.2520e-1 * t44 * t46 * t48))
t63 = lax_cond(t13 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t33 * t59 / 0.9e1)
t68 = lax_cond(t26, t7, -t9)
t69 = lax_cond(t28, t6, t68)
t72 = ((0.1e1 + t69) * r0) ** (0.1e1 / 0.3e1)
t78 = lax_cond(t27 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t72 * t59 / 0.9e1)
t79 = t63 + t78
t81 = lax_cond(t79 == 0.0e0, DBL_EPSILON, t79)
t85 = t81 ** 2
t86 = t85 ** 2
res = lax_cond(t5, 0, -0.25000000000000000000e0 * (0.1e1 - t10) * r0 * (0.360115380e1 / t81 + 0.5764e0) / (0.313901240307210000e2 / t86 + 0.149643497914092000e2 / t85 / t81 + 0.1783335908700e1 / t85))
