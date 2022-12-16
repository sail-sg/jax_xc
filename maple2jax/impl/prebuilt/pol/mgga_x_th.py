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
t27 = 0.1e1 / tau0
t30 = r0 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t40 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t42 = 4 ** (0.1e1 / 0.3e1)
t43 = 0.1e1 / t40 * t42
t47 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.27e2 / 0.80e2 * t3 * t24 * t26 * t27 * t31 * r0 * (0.1e1 + 0.7e1 / 0.216e3 * s0 / r0 * t27) * t43)
t49 = jnp.where(t8, t13, -t15)
t50 = jnp.where(t12, t9, t49)
t51 = 0.1e1 + t50
t53 = t51 ** (0.1e1 / 0.3e1)
t55 = jnp.where(t51 <= p_a_zeta_threshold, t21, t53 * t51)
t57 = 0.1e1 / tau1
t60 = r1 ** (0.1e1 / 0.3e1)
t61 = t60 ** 2
t72 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.27e2 / 0.80e2 * t3 * t55 * t26 * t57 * t61 * r1 * (0.1e1 + 0.7e1 / 0.216e3 * s2 / r1 * t57) * t43)
res = t47 + t72
