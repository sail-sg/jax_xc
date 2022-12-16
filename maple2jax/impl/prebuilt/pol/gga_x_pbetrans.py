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
t29 = math.pi ** 2
t30 = t29 ** (0.1e1 / 0.3e1)
t31 = t2 * t30
t32 = 6 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = t33 / t30
t36 = math.sqrt(s0)
t37 = r0 ** (0.1e1 / 0.3e1)
t46 = math.exp(-0.2e1 * t31 * (t35 * t36 / t37 / r0 / 0.12e2 - 0.3e1))
t49 = 0.413e0 / (0.1e1 + t46)
t50 = 0.1227e1 - t49
t51 = t30 ** 2
t53 = t32 / t51
t54 = r0 ** 2
t55 = t37 ** 2
t70 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t50 * (0.1e1 - t50 / (0.1227e1 - t49 + 0.91249999999999999998e-2 * t53 * s0 / t55 / t54))))
t72 = jnp.where(t10, t15, -t17)
t73 = jnp.where(t14, t11, t72)
t74 = 0.1e1 + t73
t76 = t74 ** (0.1e1 / 0.3e1)
t78 = jnp.where(t74 <= p_a_zeta_threshold, t23, t76 * t74)
t80 = math.sqrt(s2)
t81 = r1 ** (0.1e1 / 0.3e1)
t90 = math.exp(-0.2e1 * t31 * (t35 * t80 / t81 / r1 / 0.12e2 - 0.3e1))
t93 = 0.413e0 / (0.1e1 + t90)
t94 = 0.1227e1 - t93
t95 = r1 ** 2
t96 = t81 ** 2
t111 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t78 * t27 * (0.1e1 + t94 * (0.1e1 - t94 / (0.1227e1 - t93 + 0.91249999999999999998e-2 * t53 * s2 / t96 / t95))))
res = t70 + t111
