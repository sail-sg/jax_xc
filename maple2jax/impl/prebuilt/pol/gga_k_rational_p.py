t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = jnp.where(t15, t16, t18)
t20 = jnp.where(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = jnp.where(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t35 = 6 ** (0.1e1 / 0.3e1)
t36 = params_a_C2 / params_a_p * t35
t37 = math.pi ** 2
t38 = t37 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t40 = 0.1e1 / t39
t42 = r0 ** 2
t43 = r0 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t51 = (0.1e1 + t36 * t40 * s0 / t44 / t42 / 0.24e2) ** (-params_a_p)
t55 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * t51)
t57 = jnp.where(t11, t16, -t18)
t58 = jnp.where(t15, t12, t57)
t59 = 0.1e1 + t58
t61 = t59 ** (0.1e1 / 0.3e1)
t62 = t61 ** 2
t64 = jnp.where(t59 <= p_a_zeta_threshold, t25, t62 * t59)
t67 = r1 ** 2
t68 = r1 ** (0.1e1 / 0.3e1)
t69 = t68 ** 2
t76 = (0.1e1 + t36 * t40 * s2 / t69 / t67 / 0.24e2) ** (-params_a_p)
t80 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t64 * t31 * t76)
res = t55 + t80
