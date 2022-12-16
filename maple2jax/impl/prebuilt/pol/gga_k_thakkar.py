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
t33 = r0 ** 2
t34 = r0 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t39 = math.sqrt(s0)
t41 = 0.1e1 / t34 / r0
t42 = t39 * t41
t43 = math.asinh(t42)
t50 = 4 ** (0.1e1 / 0.3e1)
t62 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + 0.55e-2 * s0 / t35 / t33 / (0.1e1 + 0.253e-1 * t42 * t43) - 0.72e-1 * t42 / (0.2e1 * t50 * t39 * t41 + 0.1e1)))
t64 = jnp.where(t11, t16, -t18)
t65 = jnp.where(t15, t12, t64)
t66 = 0.1e1 + t65
t68 = t66 ** (0.1e1 / 0.3e1)
t69 = t68 ** 2
t71 = jnp.where(t66 <= p_a_zeta_threshold, t25, t69 * t66)
t73 = r1 ** 2
t74 = r1 ** (0.1e1 / 0.3e1)
t75 = t74 ** 2
t79 = math.sqrt(s2)
t81 = 0.1e1 / t74 / r1
t82 = t79 * t81
t83 = math.asinh(t82)
t101 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t71 * t31 * (0.1e1 + 0.55e-2 * s2 / t75 / t73 / (0.1e1 + 0.253e-1 * t82 * t83) - 0.72e-1 * t82 / (0.2e1 * t50 * t79 * t81 + 0.1e1)))
res = t62 + t101
