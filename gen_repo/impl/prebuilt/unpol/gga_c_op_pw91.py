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
t36 = 6 ** (0.1e1 / 0.3e1)
t37 = math.pi ** 2
t38 = t37 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t40 = 0.1e1 / t39
t42 = t25 ** 2
t44 = r0 ** 2
t45 = r0 ** (0.1e1 / 0.3e1)
t46 = t45 ** 2
t49 = s0 * t42 / t46 / t44
t52 = math.exp(-0.25e2 / 0.6e1 * t36 * t40 * t49)
t59 = t36 ** 2
t63 = s0 ** 2
t65 = t44 ** 2
t71 = 0.13888888888888888889e-4 * t59 / t38 / t37 * t63 * t25 / t45 / t65 / r0
t74 = t59 / t38
t75 = math.sqrt(s0)
t78 = 0.1e1 / t45 / r0
t84 = math.asinh(0.64963333333333333333e0 * t74 * t75 * t25 * t78)
t92 = 0.1e1 / (0.1e1 + ((0.2743e0 - 0.1508e0 * t52) * t36 * t40 * t49 / 0.24e2 - t71) / (0.1e1 + 0.16370833333333333333e-1 * t74 * t75 * t25 * t78 * t84 + t71))
t96 = lax_cond(t13 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t33 * t92 / 0.9e1)
t101 = lax_cond(t26, t7, -t9)
t102 = lax_cond(t28, t6, t101)
t105 = ((0.1e1 + t102) * r0) ** (0.1e1 / 0.3e1)
t111 = lax_cond(t27 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t105 * t92 / 0.9e1)
t112 = t96 + t111
t114 = lax_cond(t112 == 0.0e0, DBL_EPSILON, t112)
t118 = t114 ** 2
t119 = t118 ** 2
res = lax_cond(t5, 0, -0.25000000000000000000e0 * (0.1e1 - t10) * r0 * (0.360663084e1 / t114 + 0.5764e0) / (0.315815266717518096e2 / t119 + 0.150327320916243744e2 / t118 / t114 + 0.1788764629788e1 / t118))
