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
t36 = params_a_beta * t29 / t32 * t34
t37 = r0 ** 2
t38 = r0 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t40 = t39 * t37
t42 = s0 / t40
t43 = params_a_gamma * params_a_beta
t44 = math.sqrt(s0)
t47 = t44 / t38 / r0
t48 = math.asinh(t47)
t71 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t36 * t42 / (t43 * t47 * t48 + 0.1e1) / (0.1e1 + 0.2e1 * (t42 - l0 / t39 / r0) / s0 * t40)))
t73 = jnp.where(t10, t15, -t17)
t74 = jnp.where(t14, t11, t73)
t75 = 0.1e1 + t74
t77 = t75 ** (0.1e1 / 0.3e1)
t79 = jnp.where(t75 <= p_a_zeta_threshold, t23, t77 * t75)
t81 = r1 ** 2
t82 = r1 ** (0.1e1 / 0.3e1)
t83 = t82 ** 2
t84 = t83 * t81
t86 = s2 / t84
t87 = math.sqrt(s2)
t90 = t87 / t82 / r1
t91 = math.asinh(t90)
t114 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t79 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t36 * t86 / (t43 * t90 * t91 + 0.1e1) / (0.1e1 + 0.2e1 * (t86 - l1 / t83 / r1) / s2 * t84)))
res = t71 + t114
