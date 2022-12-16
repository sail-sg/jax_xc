t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = jnp.where(t14, t15, t17)
t19 = jnp.where(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = jnp.where(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t40 = 0.1e1 / t38 / t36
t41 = t29 ** 2
t43 = t41 / t31
t44 = math.sqrt(s0)
t51 = (0.1e1 + t43 * t44 / t37 / r0 / 0.12e2) ** 2
t52 = 0.1e1 / t51
t66 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.10008e1 + t34 * s0 * t40 * t52 * (0.1926e0 + 0.79008333333333333333e-1 * t34 * s0 * t40 * t52) / 0.24e2))
t68 = jnp.where(t10, t15, -t17)
t69 = jnp.where(t14, t11, t68)
t70 = 0.1e1 + t69
t72 = t70 ** (0.1e1 / 0.3e1)
t74 = jnp.where(t70 <= p_a_zeta_threshold, t23, t72 * t70)
t77 = r1 ** 2
t78 = r1 ** (0.1e1 / 0.3e1)
t79 = t78 ** 2
t81 = 0.1e1 / t79 / t77
t82 = math.sqrt(s2)
t89 = (0.1e1 + t43 * t82 / t78 / r1 / 0.12e2) ** 2
t90 = 0.1e1 / t89
t104 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t74 * t27 * (0.10008e1 + t34 * s2 * t81 * t90 * (0.1926e0 + 0.79008333333333333333e-1 * t34 * s2 * t81 * t90) / 0.24e2))
res = t66 + t104
