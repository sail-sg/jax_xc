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
t34 = r0 ** 2
t35 = r0 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t41 = math.sqrt(s0)
t44 = 0.1e1 / t35 / r0
t45 = 2 ** (0.1e1 / 0.3e1)
t56 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (params_a_aa + 0.13888888888888888889e-1 * params_a_bb * s0 / t36 / t34 + params_a_cc * t41 * t44 / (0.4e1 * t41 * t44 + t45)))
t58 = jnp.where(t11, t16, -t18)
t59 = jnp.where(t15, t12, t58)
t60 = 0.1e1 + t59
t62 = t60 ** (0.1e1 / 0.3e1)
t63 = t62 ** 2
t65 = jnp.where(t60 <= p_a_zeta_threshold, t25, t63 * t60)
t68 = r1 ** 2
t69 = r1 ** (0.1e1 / 0.3e1)
t70 = t69 ** 2
t75 = math.sqrt(s2)
t78 = 0.1e1 / t69 / r1
t89 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t65 * t31 * (params_a_aa + 0.13888888888888888889e-1 * params_a_bb * s2 / t70 / t68 + params_a_cc * t75 * t78 / (0.4e1 * t75 * t78 + t45)))
res = t56 + t89
