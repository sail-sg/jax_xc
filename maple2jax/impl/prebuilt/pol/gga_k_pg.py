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
t33 = 6 ** (0.1e1 / 0.3e1)
t34 = math.pi ** 2
t35 = t34 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t37 = 0.1e1 / t36
t38 = t33 * t37
t39 = r0 ** 2
t40 = r0 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t43 = 0.1e1 / t41 / t39
t47 = params_a_pg_mu * t33
t52 = math.exp(-t47 * t37 * s0 * t43 / 0.24e2)
t57 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.5e1 / 0.72e2 * t38 * s0 * t43 + t52))
t59 = jnp.where(t11, t16, -t18)
t60 = jnp.where(t15, t12, t59)
t61 = 0.1e1 + t60
t63 = t61 ** (0.1e1 / 0.3e1)
t64 = t63 ** 2
t66 = jnp.where(t61 <= p_a_zeta_threshold, t25, t64 * t61)
t68 = r1 ** 2
t69 = r1 ** (0.1e1 / 0.3e1)
t70 = t69 ** 2
t72 = 0.1e1 / t70 / t68
t80 = math.exp(-t47 * t37 * s2 * t72 / 0.24e2)
t85 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t66 * t31 * (0.5e1 / 0.72e2 * t38 * s2 * t72 + t80))
res = t57 + t85
