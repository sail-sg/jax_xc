t2 = r0 + r1
t3 = 0.1e1 / t2
t4 = (r0 - r1) * t3
t5 = abs(t4)
t11 = jnp.logical_or(0.1e1 - t5 <= p_a_zeta_threshold, r0 <= p_a_dens_threshold and r1 <= p_a_dens_threshold)
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
t40 = t36 / t38
t41 = 4 ** (0.1e1 / 0.3e1)
t42 = 2 ** (0.1e1 / 0.3e1)
t43 = t41 * t42
t44 = t31 <= p_a_zeta_threshold
t45 = 0.1e1 - t30
t46 = t45 <= p_a_zeta_threshold
t47 = jnp.where(t46, t17, t30)
t48 = jnp.where(t44, t14, t47)
t51 = ((0.1e1 + t48) * t2) ** (0.1e1 / 0.3e1)
t56 = jnp.where(t31 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t40 * t43 / t51 / 0.9e1)
t61 = jnp.where(t44, t17, -t30)
t62 = jnp.where(t46, t14, t61)
t65 = ((0.1e1 + t62) * t2) ** (0.1e1 / 0.3e1)
t70 = jnp.where(t45 * t2 / 0.2e1 <= p_a_dens_threshold, 0, t40 * t43 / t65 / 0.9e1)
t71 = t56 + t70
t73 = jnp.where(t71 == 0.0e0, DBL_EPSILON, t71)
t77 = t73 ** 2
t78 = t77 ** 2
res = jnp.where(t11, 0, -0.25000000000000000000e0 * (0.1e1 - t20) * t2 * (0.390299956e1 / t73 + 0.5764e0) / (0.433132090567376656e2 / t78 + 0.190514637481962976e2 / t77 / t73 + 0.2094820520028e1 / t77))
