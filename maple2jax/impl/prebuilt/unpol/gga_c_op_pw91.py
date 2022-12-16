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
t35 = 6 ** (0.1e1 / 0.3e1)
t36 = math.pi ** 2
t37 = t36 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t39 = 0.1e1 / t38
t41 = t24 ** 2
t43 = r0 ** 2
t44 = r0 ** (0.1e1 / 0.3e1)
t45 = t44 ** 2
t48 = s0 * t41 / t45 / t43
t51 = math.exp(-0.25e2 / 0.6e1 * t35 * t39 * t48)
t58 = t35 ** 2
t62 = s0 ** 2
t64 = t43 ** 2
t70 = 0.13888888888888888889e-4 * t58 / t37 / t36 * t62 * t24 / t44 / t64 / r0
t73 = t58 / t37
t74 = math.sqrt(s0)
t77 = 0.1e1 / t44 / r0
t83 = math.asinh(0.64963333333333333333e0 * t73 * t74 * t24 * t77)
t91 = 0.1e1 / (0.1e1 + ((0.2743e0 - 0.1508e0 * t51) * t35 * t39 * t48 / 0.24e2 - t70) / (0.1e1 + 0.16370833333333333333e-1 * t73 * t74 * t24 * t77 * t83 + t70))
t95 = jnp.where(t12 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t32 * t91 / 0.9e1)
t100 = jnp.where(t25, t6, -t8)
t101 = jnp.where(t27, t5, t100)
t104 = ((0.1e1 + t101) * r0) ** (0.1e1 / 0.3e1)
t110 = jnp.where(t26 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t104 * t91 / 0.9e1)
t111 = t95 + t110
t113 = jnp.where(t111 == 0.0e0, DBL_EPSILON, t111)
t117 = t113 ** 2
t118 = t117 ** 2
res = jnp.where(t4, 0, -0.25000000000000000000e0 * (0.1e1 - t9) * r0 * (0.360663084e1 / t113 + 0.5764e0) / (0.315815266717518096e2 / t118 + 0.150327320916243744e2 / t117 / t113 + 0.1788764629788e1 / t117))
