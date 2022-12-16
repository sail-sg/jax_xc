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
t41 = t24 ** 2
t43 = r0 ** 2
t44 = r0 ** (0.1e1 / 0.3e1)
t45 = t44 ** 2
t55 = 0.1e1 / (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t35 / t38 * s0 * t41 / t45 / t43))
t59 = jnp.where(t12 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t32 * t55 / 0.9e1)
t64 = jnp.where(t25, t6, -t8)
t65 = jnp.where(t27, t5, t64)
t68 = ((0.1e1 + t65) * r0) ** (0.1e1 / 0.3e1)
t74 = jnp.where(t26 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t68 * t55 / 0.9e1)
t75 = t59 + t74
t77 = jnp.where(t75 == 0.0e0, DBL_EPSILON, t75)
t81 = t77 ** 2
t82 = t81 ** 2
res = jnp.where(t4, 0, -0.25000000000000000000e0 * (0.1e1 - t9) * r0 * (0.361925846e1 / t77 + 0.5764e0) / (0.320261508740743441e2 / t82 + 0.151911844324290596e2 / t81 / t77 + 0.1801312286343e1 / t81))
