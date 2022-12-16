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
t30 = t29 ** 2
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = 0.1e1 / t32
t34 = t30 * t33
t35 = math.sqrt(s0)
t36 = r0 ** (0.1e1 / 0.3e1)
t38 = 0.1e1 / t36 / r0
t39 = t35 * t38
t41 = t34 * t39 / 0.12e2
t42 = math.tanh(t41)
t44 = math.asinh(t41)
t47 = (0.1e1 - params_a_zeta) * t30 * t33
t49 = math.log(0.1e1 + t41)
t52 = params_a_zeta * t30
t71 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params_a_mu * t42 * t44 * (0.1e1 + params_a_alpha * (t52 * t33 * t35 * t38 / 0.12e2 + t47 * t39 * t49 / 0.12e2)) / (params_a_beta * t42 * t44 + 0.1e1)))
t73 = jnp.where(t10, t15, -t17)
t74 = jnp.where(t14, t11, t73)
t75 = 0.1e1 + t74
t77 = t75 ** (0.1e1 / 0.3e1)
t79 = jnp.where(t75 <= p_a_zeta_threshold, t23, t77 * t75)
t81 = math.sqrt(s2)
t82 = r1 ** (0.1e1 / 0.3e1)
t84 = 0.1e1 / t82 / r1
t85 = t81 * t84
t87 = t34 * t85 / 0.12e2
t88 = math.tanh(t87)
t90 = math.asinh(t87)
t92 = math.log(0.1e1 + t87)
t113 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t79 * t27 * (0.1e1 + params_a_mu * t88 * t90 * (0.1e1 + params_a_alpha * (t52 * t33 * t81 * t84 / 0.12e2 + t47 * t85 * t92 / 0.12e2)) / (params_a_beta * t88 * t90 + 0.1e1)))
res = t71 + t113
