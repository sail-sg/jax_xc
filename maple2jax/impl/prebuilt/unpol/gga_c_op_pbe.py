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
t42 = t25 ** 2
t44 = r0 ** 2
t45 = r0 ** (0.1e1 / 0.3e1)
t46 = t45 ** 2
t56 = 0.1e1 / (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t36 / t39 * s0 * t42 / t46 / t44))
t60 = lax_cond(t13 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t33 * t56 / 0.9e1)
t65 = lax_cond(t26, t7, -t9)
t66 = lax_cond(t28, t6, t65)
t69 = ((0.1e1 + t66) * r0) ** (0.1e1 / 0.3e1)
t75 = lax_cond(t27 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t69 * t56 / 0.9e1)
t76 = t60 + t75
t78 = lax_cond(t76 == 0.0e0, DBL_EPSILON, t76)
t82 = t78 ** 2
t83 = t82 ** 2
res = lax_cond(t5, 0, -0.25000000000000000000e0 * (0.1e1 - t10) * r0 * (0.361925846e1 / t78 + 0.5764e0) / (0.320261508740743441e2 / t83 + 0.151911844324290596e2 / t82 / t78 + 0.1801312286343e1 / t82))
