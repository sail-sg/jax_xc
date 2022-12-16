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
t30 = r0 ** 2
t31 = r0 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = 0.1e1 / t32 / t30
t38 = (params_a_gamma * s0 * t34 + 0.1e1) ** params_a_omega
t46 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params_a_beta * s0 * t34 / t38))
t48 = jnp.where(t10, t15, -t17)
t49 = jnp.where(t14, t11, t48)
t50 = 0.1e1 + t49
t52 = t50 ** (0.1e1 / 0.3e1)
t54 = jnp.where(t50 <= p_a_zeta_threshold, t23, t52 * t50)
t57 = r1 ** 2
t58 = r1 ** (0.1e1 / 0.3e1)
t59 = t58 ** 2
t61 = 0.1e1 / t59 / t57
t65 = (params_a_gamma * s2 * t61 + 0.1e1) ** params_a_omega
t73 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t54 * t27 * (0.1e1 + params_a_beta * s2 * t61 / t65))
res = t46 + t73
