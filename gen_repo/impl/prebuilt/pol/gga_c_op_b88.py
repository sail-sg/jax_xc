t2 = r0 + r1
t3 = 0.1e1 / t2
t4 = (r0 - r1) * t3
t5 = abs(t4)
t10 = jnp.logical_and(r0 <= p_a_dens_threshold, r1 <= p_a_dens_threshold)
t11 = jnp.logical_or(0.1e1 - t5 <= p_a_zeta_threshold, t10)
t14 = p_a_zeta_threshold - 0.1e1
t17 = -t14
t18 = lax_cond(0.1e1 - t4 <= p_a_zeta_threshold, t17, t4)
t19 = lax_cond(0.1e1 + t4 <= p_a_zeta_threshold, t14, t18)
t20 = t19 ** 2
t29 = lax_cond(0.2e1 * r1 * t3 <= p_a_zeta_threshold, t17, t4)
t30 = lax_cond(0.2e1 * r0 * t3 <= p_a_zeta_threshold, t14, t29)
t31 = 0.1e1 + t30
t35 = 3 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t41 = 4 ** (0.1e1 / 0.3e1)
t42 = t36 / t38 * t41
t43 = 2 ** (0.1e1 / 0.3e1)
t44 = t31 <= p_a_zeta_threshold
t45 = 0.1e1 - t30
t46 = t45 <= p_a_zeta_threshold
t47 = lax_cond(t46, t17, t30)
t48 = lax_cond(t44, t14, t47)
t51 = ((0.1e1 + t48) * t2) ** (0.1e1 / 0.3e1)
t54 = r0 ** 2
t55 = r0 ** (0.1e1 / 0.3e1)
t56 = t55 ** 2
t60 = math.sqrt(s0)
t63 = t60 / t55 / r0
t64 = math.asinh(t63)
t77 = lax_cond(t31 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t51 / (0.1e1 + 0.93333333333333333332e-3 * t42 * s0 / t56 / t54 / (0.1e1 + 0.2520e-1 * t63 * t64)) / 0.9e1)
t82 = lax_cond(t44, t17, -t30)
t83 = lax_cond(t46, t14, t82)
t86 = ((0.1e1 + t83) * t2) ** (0.1e1 / 0.3e1)
t89 = r1 ** 2
t90 = r1 ** (0.1e1 / 0.3e1)
t91 = t90 ** 2
t95 = math.sqrt(s2)
t98 = t95 / t90 / r1
t99 = math.asinh(t98)
t112 = lax_cond(t45 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t86 / (0.1e1 + 0.93333333333333333332e-3 * t42 * s2 / t91 / t89 / (0.1e1 + 0.2520e-1 * t98 * t99)) / 0.9e1)
t113 = t77 + t112
t115 = lax_cond(t113 == 0.0e0, DBL_EPSILON, t113)
t119 = t115 ** 2
t120 = t119 ** 2
res = lax_cond(t11, 0, -0.25000000000000000000e0 * (0.1e1 - t20) * t2 * (0.360115380e1 / t115 + 0.5764e0) / (0.313901240307210000e2 / t120 + 0.149643497914092000e2 / t119 / t115 + 0.1783335908700e1 / t119))
