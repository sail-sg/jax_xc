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
t21 = t17 / t19
t22 = 4 ** (0.1e1 / 0.3e1)
t23 = 2 ** (0.1e1 / 0.3e1)
t24 = t22 * t23
t25 = t12 <= p_a_zeta_threshold
t26 = 0.1e1 - t8
t27 = t26 <= p_a_zeta_threshold
t28 = jnp.where(t27, t6, t8)
t29 = jnp.where(t25, t5, t28)
t32 = ((0.1e1 + t29) * r0) ** (0.1e1 / 0.3e1)
t37 = jnp.where(t12 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t21 * t24 / t32 / 0.9e1)
t42 = jnp.where(t25, t6, -t8)
t43 = jnp.where(t27, t5, t42)
t46 = ((0.1e1 + t43) * r0) ** (0.1e1 / 0.3e1)
t51 = jnp.where(t26 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t21 * t24 / t46 / 0.9e1)
t52 = t37 + t51
t54 = jnp.where(t52 == 0.0e0, DBL_EPSILON, t52)
t58 = t54 ** 2
t59 = t58 ** 2
res = jnp.where(t4, 0, -0.25000000000000000000e0 * (0.1e1 - t9) * r0 * (0.390299956e1 / t54 + 0.5764e0) / (0.433132090567376656e2 / t59 + 0.190514637481962976e2 / t58 / t54 + 0.2094820520028e1 / t58))
