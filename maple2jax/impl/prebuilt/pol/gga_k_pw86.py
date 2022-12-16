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
t33 = 6 ** (0.1e1 / 0.3e1)
t34 = math.pi ** 2
t35 = t34 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = t33 / t36
t39 = r0 ** 2
t40 = r0 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t47 = t33 ** 2
t50 = t47 / t35 / t34
t51 = s0 ** 2
t52 = t39 ** 2
t59 = t34 ** 2
t60 = 0.1e1 / t59
t63 = t52 ** 2
t68 = (0.1e1 + 0.91999999999999999998e-1 * t38 * s0 / t41 / t39 + 0.16093750000000000000e-1 * t50 * t51 / t40 / t52 / r0 + 0.86805555555555555555e-4 * t60 * t51 * s0 / t63) ** (0.1e1 / 0.15e2)
t72 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * t68)
t74 = jnp.where(t11, t16, -t18)
t75 = jnp.where(t15, t12, t74)
t76 = 0.1e1 + t75
t78 = t76 ** (0.1e1 / 0.3e1)
t79 = t78 ** 2
t81 = jnp.where(t76 <= p_a_zeta_threshold, t25, t79 * t76)
t83 = r1 ** 2
t84 = r1 ** (0.1e1 / 0.3e1)
t85 = t84 ** 2
t91 = s2 ** 2
t92 = t83 ** 2
t101 = t92 ** 2
t106 = (0.1e1 + 0.91999999999999999998e-1 * t38 * s2 / t85 / t83 + 0.16093750000000000000e-1 * t50 * t91 / t84 / t92 / r1 + 0.86805555555555555555e-4 * t60 * t91 * s2 / t101) ** (0.1e1 / 0.15e2)
t110 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t81 * t31 * t106)
res = t72 + t110
