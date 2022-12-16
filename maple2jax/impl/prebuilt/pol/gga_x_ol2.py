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
t37 = math.sqrt(s0)
t40 = 0.1e1 / t31 / r0
t41 = 2 ** (0.1e1 / 0.3e1)
t52 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (params_a_aa + 0.13888888888888888889e-1 * params_a_bb * s0 / t32 / t30 + params_a_cc * t37 * t40 / (0.4e1 * t37 * t40 + t41)))
t54 = jnp.where(t10, t15, -t17)
t55 = jnp.where(t14, t11, t54)
t56 = 0.1e1 + t55
t58 = t56 ** (0.1e1 / 0.3e1)
t60 = jnp.where(t56 <= p_a_zeta_threshold, t23, t58 * t56)
t63 = r1 ** 2
t64 = r1 ** (0.1e1 / 0.3e1)
t65 = t64 ** 2
t70 = math.sqrt(s2)
t73 = 0.1e1 / t64 / r1
t84 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t60 * t27 * (params_a_aa + 0.13888888888888888889e-1 * params_a_bb * s2 / t65 / t63 + params_a_cc * t70 * t73 / (0.4e1 * t70 * t73 + t41)))
res = t52 + t84
