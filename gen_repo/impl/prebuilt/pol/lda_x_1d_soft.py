t3 = r0 + r1
t4 = 0.1e1 / t3
t5 = (r0 - r1) * t4
t7 = 0.1e1 + t5 <= p_a_zeta_threshold
t8 = jnp.logical_or(r0 <= p_a_dens_threshold, t7)
t9 = p_a_zeta_threshold - 0.1e1
t11 = 0.1e1 - t5 <= p_a_zeta_threshold
t12 = -t9
t13 = lax_cond(t11, t12, t5)
t14 = lax_cond(t7, t9, t13)
t15 = 0.1e1 + t14
t17 = params_a_beta * t3
t18 = t15 * math.pi * t17
t19 = int1(t18)
t21 = int2(t18)
t22 = 0.1e1 / math.pi
t24 = 0.1e1 / params_a_beta
t25 = t24 * t4
t31 = lax_cond(t8, 0, -0.25000000000000000000e0 * (-t21 * t22 * t25 + t15 * t19) * t22 * t24)
t33 = jnp.logical_or(r1 <= p_a_dens_threshold, t11)
t34 = lax_cond(t7, t12, -t5)
t35 = lax_cond(t11, t9, t34)
t36 = 0.1e1 + t35
t38 = t36 * math.pi * t17
t39 = int1(t38)
t41 = int2(t38)
t48 = lax_cond(t33, 0, -0.25000000000000000000e0 * (-t41 * t22 * t25 + t36 * t39) * t22 * t24)
res = t31 + t48
