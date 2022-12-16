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
t31 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t33 = t29 / t31
t34 = 4 ** (0.1e1 / 0.3e1)
t35 = math.sqrt(s0)
t36 = r0 ** (0.1e1 / 0.3e1)
t39 = t35 / t36 / r0
t40 = math.sqrt(t39)
t49 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + 0.2e1 / 0.1233e4 * t33 * t34 * t40 * t39))
t51 = jnp.where(t10, t15, -t17)
t52 = jnp.where(t14, t11, t51)
t53 = 0.1e1 + t52
t55 = t53 ** (0.1e1 / 0.3e1)
t57 = jnp.where(t53 <= p_a_zeta_threshold, t23, t55 * t53)
t59 = math.sqrt(s2)
t60 = r1 ** (0.1e1 / 0.3e1)
t63 = t59 / t60 / r1
t64 = math.sqrt(t63)
t73 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t57 * t27 * (0.1e1 + 0.2e1 / 0.1233e4 * t33 * t34 * t64 * t63))
res = t49 + t73
