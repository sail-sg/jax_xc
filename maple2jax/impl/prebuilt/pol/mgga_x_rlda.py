t2 = math.pi ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
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
t20 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t21 = t20 * p_a_zeta_threshold
t22 = t18 ** (0.1e1 / 0.3e1)
t24 = jnp.where(t18 <= p_a_zeta_threshold, t21, t22 * t18)
t26 = t4 ** (0.1e1 / 0.3e1)
t29 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t31 = params_a_prefactor / t29
t32 = 4 ** (0.1e1 / 0.3e1)
t33 = r0 ** (0.1e1 / 0.3e1)
t34 = t33 ** 2
t36 = 0.1e1 / t34 / r0
t47 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.15e2 / 0.16e2 * t3 * t24 * t26 * t31 * t32 / (0.2e1 * tau0 * t36 - l0 * t36 / 0.4e1))
t49 = jnp.where(t8, t13, -t15)
t50 = jnp.where(t12, t9, t49)
t51 = 0.1e1 + t50
t53 = t51 ** (0.1e1 / 0.3e1)
t55 = jnp.where(t51 <= p_a_zeta_threshold, t21, t53 * t51)
t58 = r1 ** (0.1e1 / 0.3e1)
t59 = t58 ** 2
t61 = 0.1e1 / t59 / r1
t72 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.15e2 / 0.16e2 * t3 * t55 * t26 * t31 * t32 / (0.2e1 * tau1 * t61 - l1 * t61 / 0.4e1))
res = t47 + t72
