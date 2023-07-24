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
t22 = t18 / t20
t23 = 4 ** (0.1e1 / 0.3e1)
t24 = t22 * t23
t25 = 2 ** (0.1e1 / 0.3e1)
t26 = t13 <= p_a_zeta_threshold
t27 = 0.1e1 - t9
t28 = t27 <= p_a_zeta_threshold
t29 = lax_cond(t28, t7, t9)
t30 = lax_cond(t26, t6, t29)
t33 = ((0.1e1 + t30) * r0) ** (0.1e1 / 0.3e1)
t36 = math.sqrt(s0)
t38 = r0 ** (0.1e1 / 0.3e1)
t41 = t36 * t25 / t38 / r0
t42 = math.sqrt(t41)
t48 = 0.1e1 / (0.1e1 + 0.2e1 / 0.1233e4 * t22 * t23 * t42 * t41)
t52 = lax_cond(t13 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t33 * t48 / 0.9e1)
t57 = lax_cond(t26, t7, -t9)
t58 = lax_cond(t28, t6, t57)
t61 = ((0.1e1 + t58) * r0) ** (0.1e1 / 0.3e1)
t67 = lax_cond(t27 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t24 * t25 / t61 * t48 / 0.9e1)
t68 = t52 + t67
t70 = lax_cond(t68 == 0.0e0, DBL_EPSILON, t68)
t74 = t70 ** 2
t75 = t74 ** 2
res = lax_cond(t5, 0, -0.25000000000000000000e0 * (0.1e1 - t10) * r0 * (0.359628532e1 / t70 + 0.5764e0) / (0.312207199195441936e2 / t75 + 0.149037398922132448e2 / t74 / t70 + 0.1778517305052e1 / t74))
