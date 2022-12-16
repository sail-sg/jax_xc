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
t23 = t21 * t22
t24 = 2 ** (0.1e1 / 0.3e1)
t25 = t12 <= p_a_zeta_threshold
t26 = 0.1e1 - t8
t27 = t26 <= p_a_zeta_threshold
t28 = jnp.where(t27, t6, t8)
t29 = jnp.where(t25, t5, t28)
t32 = ((0.1e1 + t29) * r0) ** (0.1e1 / 0.3e1)
t35 = math.sqrt(s0)
t37 = r0 ** (0.1e1 / 0.3e1)
t40 = t35 * t24 / t37 / r0
t41 = math.sqrt(t40)
t47 = 0.1e1 / (0.1e1 + 0.2e1 / 0.1233e4 * t21 * t22 * t41 * t40)
t51 = jnp.where(t12 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t32 * t47 / 0.9e1)
t56 = jnp.where(t25, t6, -t8)
t57 = jnp.where(t27, t5, t56)
t60 = ((0.1e1 + t57) * r0) ** (0.1e1 / 0.3e1)
t66 = jnp.where(t26 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t23 * t24 / t60 * t47 / 0.9e1)
t67 = t51 + t66
t69 = jnp.where(t67 == 0.0e0, DBL_EPSILON, t67)
t73 = t69 ** 2
t74 = t73 ** 2
res = jnp.where(t4, 0, -0.25000000000000000000e0 * (0.1e1 - t9) * r0 * (0.359628532e1 / t69 + 0.5764e0) / (0.312207199195441936e2 / t74 + 0.149037398922132448e2 / t73 / t69 + 0.1778517305052e1 / t73))
