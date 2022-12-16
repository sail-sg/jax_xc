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
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t32 = math.pi ** 2
t33 = t32 ** (0.1e1 / 0.3e1)
t34 = 0.1e1 / t33
t35 = params_a_B1 * t30 * t34
t36 = math.sqrt(s0)
t37 = r0 ** (0.1e1 / 0.3e1)
t40 = t36 / t37 / r0
t41 = t30 * t34
t45 = math.log(0.1e1 + t41 * t40 / 0.12e2)
t50 = params_a_B2 * t30 * t34
t52 = math.log(0.1e1 + t45)
t60 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + t35 * t40 * t45 / 0.12e2 + t50 * t40 * t52 / 0.12e2))
t62 = jnp.where(t10, t15, -t17)
t63 = jnp.where(t14, t11, t62)
t64 = 0.1e1 + t63
t66 = t64 ** (0.1e1 / 0.3e1)
t68 = jnp.where(t64 <= p_a_zeta_threshold, t23, t66 * t64)
t70 = math.sqrt(s2)
t71 = r1 ** (0.1e1 / 0.3e1)
t74 = t70 / t71 / r1
t78 = math.log(0.1e1 + t41 * t74 / 0.12e2)
t83 = math.log(0.1e1 + t78)
t91 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t68 * t27 * (0.1e1 + t35 * t74 * t78 / 0.12e2 + t50 * t74 * t83 / 0.12e2))
res = t60 + t91
