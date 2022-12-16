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
t29 = t2 ** 2
t32 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t34 = 4 ** (0.1e1 / 0.3e1)
t36 = params_a_gamma * t29 / t32 * t34
t37 = 2 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t39 = t20 * t6
t40 = t39 ** (0.1e1 / 0.3e1)
t42 = t38 * t40 * t39
t43 = r0 ** 2
t44 = r0 ** (0.1e1 / 0.3e1)
t45 = t44 ** 2
t60 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - t36 * t42 * s0 / t45 / t43 / (t42 / 0.4e1 + params_a_delta) / 0.18e2))
t62 = jnp.where(t10, t15, -t17)
t63 = jnp.where(t14, t11, t62)
t64 = 0.1e1 + t63
t66 = t64 ** (0.1e1 / 0.3e1)
t68 = jnp.where(t64 <= p_a_zeta_threshold, t23, t66 * t64)
t70 = t64 * t6
t71 = t70 ** (0.1e1 / 0.3e1)
t73 = t38 * t71 * t70
t74 = r1 ** 2
t75 = r1 ** (0.1e1 / 0.3e1)
t76 = t75 ** 2
t91 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 - t36 * t73 * s2 / t76 / t74 / (t73 / 0.4e1 + params_a_delta) / 0.18e2))
res = t60 + t91
