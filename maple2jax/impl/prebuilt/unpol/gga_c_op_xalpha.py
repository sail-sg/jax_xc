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
t24 = 2 ** (0.1e1 / 0.3e1)
t25 = t23 * t24
t26 = t13 <= p_a_zeta_threshold
t27 = 0.1e1 - t9
t28 = t27 <= p_a_zeta_threshold
t29 = lax_cond(t28, t7, t9)
t30 = lax_cond(t26, t6, t29)
t33 = ((0.1e1 + t30) * r0) ** (0.1e1 / 0.3e1)
t38 = lax_cond(t13 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t22 * t25 / t33 / 0.9e1)
t43 = lax_cond(t26, t7, -t9)
t44 = lax_cond(t28, t6, t43)
t47 = ((0.1e1 + t44) * r0) ** (0.1e1 / 0.3e1)
t52 = lax_cond(t27 * r0 / 0.2e1 <= p_a_dens_threshold, 0, t22 * t25 / t47 / 0.9e1)
t53 = t38 + t52
t55 = lax_cond(t53 == 0.0e0, DBL_EPSILON, t53)
t59 = t55 ** 2
t60 = t59 ** 2
res = lax_cond(t5, 0, -0.25000000000000000000e0 * (0.1e1 - t10) * r0 * (0.390299956e1 / t55 + 0.5764e0) / (0.433132090567376656e2 / t60 + 0.190514637481962976e2 / t59 / t55 + 0.2094820520028e1 / t59))
