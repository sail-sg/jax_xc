t2 = math.sqrt(math.pi)
t3 = 0.1e1 / t2
t4 = r0 + r1
t5 = 0.1e1 / t4
t8 = 0.2e1 * r0 * t5 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t12 = 0.2e1 * r1 * t5 <= p_a_zeta_threshold
t13 = -t9
t15 = (r0 - r1) * t5
t16 = jnp.where(t12, t13, t15)
t17 = jnp.where(t8, t9, t16)
t18 = 0.1e1 + t17
t20 = math.sqrt(p_a_zeta_threshold)
t21 = t20 * p_a_zeta_threshold
t22 = math.sqrt(t18)
t24 = jnp.where(t18 <= p_a_zeta_threshold, t21, t22 * t18)
t26 = math.sqrt(0.2e1)
t27 = math.sqrt(t4)
t28 = t26 * t27
t30 = r0 ** 2
t32 = 0.1e1 / t30 / r0
t36 = (0.1e1 + 0.8323e-2 * s0 * t32) ** (0.1e1 / 0.4e1)
t37 = t36 ** 2
t47 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t28 * (0.1e1 + 0.12438750000000000000e-2 * t2 * s0 * t32 / t37 / t36))
t49 = jnp.where(t8, t13, -t15)
t50 = jnp.where(t12, t9, t49)
t51 = 0.1e1 + t50
t53 = math.sqrt(t51)
t55 = jnp.where(t51 <= p_a_zeta_threshold, t21, t53 * t51)
t58 = r1 ** 2
t60 = 0.1e1 / t58 / r1
t64 = (0.1e1 + 0.8323e-2 * s2 * t60) ** (0.1e1 / 0.4e1)
t65 = t64 ** 2
t75 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t55 * t28 * (0.1e1 + 0.12438750000000000000e-2 * t2 * s2 * t60 / t65 / t64))
res = t47 + t75
