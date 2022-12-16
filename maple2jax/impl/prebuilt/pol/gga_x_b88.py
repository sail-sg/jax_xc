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
t34 = params_a_beta * t29 / t32
t35 = 4 ** (0.1e1 / 0.3e1)
t37 = r0 ** 2
t38 = r0 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t42 = params_a_gamma * params_a_beta
t43 = math.sqrt(s0)
t46 = t43 / t38 / r0
t47 = math.asinh(t46)
t60 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t34 * t35 * s0 / t39 / t37 / (t42 * t46 * t47 + 0.1e1)))
t62 = jnp.where(t10, t15, -t17)
t63 = jnp.where(t14, t11, t62)
t64 = 0.1e1 + t63
t66 = t64 ** (0.1e1 / 0.3e1)
t68 = jnp.where(t64 <= p_a_zeta_threshold, t23, t66 * t64)
t71 = r1 ** 2
t72 = r1 ** (0.1e1 / 0.3e1)
t73 = t72 ** 2
t76 = math.sqrt(s2)
t79 = t76 / t72 / r1
t80 = math.asinh(t79)
t93 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 + 0.2e1 / 0.9e1 * t34 * t35 * s2 / t73 / t71 / (t42 * t79 * t80 + 0.1e1)))
res = t60 + t93
