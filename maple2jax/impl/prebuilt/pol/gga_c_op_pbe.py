t2 = r0 + r1
t3 = 0.1e1 / t2
t4 = (r0 - r1) * t3
t5 = abs(t4)
t10 = jnp.logical_and(r0 <= p_a_dens_threshold, r1 <= p_a_dens_threshold)
t11 = jnp.logical_or(0.1e1 - t5 <= p_a_zeta_threshold, t10)
t14 = p_a_zeta_threshold - 0.1e1
t17 = -t14
t18 = jnp.where(0.1e1 - t4 <= p_a_zeta_threshold, t17, t4)
t19 = jnp.where(0.1e1 + t4 <= p_a_zeta_threshold, t14, t18)
t20 = t19 ** 2
t29 = jnp.where(0.2e1 * r1 * t3 <= p_a_zeta_threshold, t17, t4)
t30 = jnp.where(0.2e1 * r0 * t3 <= p_a_zeta_threshold, t14, t29)
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
t47 = jnp.where(t46, t17, t30)
t48 = jnp.where(t44, t14, t47)
t51 = ((0.1e1 + t48) * t2) ** (0.1e1 / 0.3e1)
t54 = 6 ** (0.1e1 / 0.3e1)
t55 = math.pi ** 2
t56 = t55 ** (0.1e1 / 0.3e1)
t57 = t56 ** 2
t59 = t54 / t57
t60 = r0 ** 2
t61 = r0 ** (0.1e1 / 0.3e1)
t62 = t61 ** 2
t76 = jnp.where(t31 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t51 / (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t59 * s0 / t62 / t60)) / 0.9e1)
t81 = jnp.where(t44, t17, -t30)
t82 = jnp.where(t46, t14, t81)
t85 = ((0.1e1 + t82) * t2) ** (0.1e1 / 0.3e1)
t88 = r1 ** 2
t89 = r1 ** (0.1e1 / 0.3e1)
t90 = t89 ** 2
t104 = jnp.where(t45 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t42 * t43 / t85 / (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91464571985215458336e-2 * t59 * s2 / t90 / t88)) / 0.9e1)
t105 = t76 + t104
t107 = jnp.where(t105 == 0.0e0, DBL_EPSILON, t105)
t111 = t107 ** 2
t112 = t111 ** 2
res = jnp.where(t11, 0, -0.25000000000000000000e0 * (0.1e1 - t20) * t2 * (0.361925846e1 / t107 + 0.5764e0) / (0.320261508740743441e2 / t112 + 0.151911844324290596e2 / t111 / t107 + 0.1801312286343e1 / t111))
