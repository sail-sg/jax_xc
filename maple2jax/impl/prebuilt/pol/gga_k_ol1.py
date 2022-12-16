t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = jnp.where(t15, t16, t18)
t20 = jnp.where(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = jnp.where(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t33 = r0 ** 2
t34 = r0 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t40 = 2 ** (0.1e1 / 0.3e1)
t41 = math.sqrt(s0)
t48 = 6 ** (0.1e1 / 0.3e1)
t50 = math.pi ** 2
t51 = t50 ** (0.1e1 / 0.3e1)
t52 = t51 ** 2
t53 = 0.1e1 / t52
t60 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + 0.5e1 / 0.9e1 * (s0 / t35 / t33 / 0.72e2 + 0.677e-2 * t40 * t41 / t34 / r0) * t48 * t53))
t62 = jnp.where(t11, t16, -t18)
t63 = jnp.where(t15, t12, t62)
t64 = 0.1e1 + t63
t66 = t64 ** (0.1e1 / 0.3e1)
t67 = t66 ** 2
t69 = jnp.where(t64 <= p_a_zeta_threshold, t25, t67 * t64)
t71 = r1 ** 2
t72 = r1 ** (0.1e1 / 0.3e1)
t73 = t72 ** 2
t78 = math.sqrt(s2)
t92 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t69 * t31 * (0.1e1 + 0.5e1 / 0.9e1 * (s2 / t73 / t71 / 0.72e2 + 0.677e-2 * t40 * t78 / t72 / r1) * t48 * t53))
res = t60 + t92
