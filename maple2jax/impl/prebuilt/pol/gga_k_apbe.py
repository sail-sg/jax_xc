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
t34 = params_a_mu * t33
t35 = math.pi ** 2
t36 = t35 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t38 = 0.1e1 / t37
t40 = r0 ** 2
t41 = r0 ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t57 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t34 * t38 * s0 / t42 / t40 / 0.24e2))))
t59 = jnp.where(t11, t16, -t18)
t60 = jnp.where(t15, t12, t59)
t61 = 0.1e1 + t60
t63 = t61 ** (0.1e1 / 0.3e1)
t64 = t63 ** 2
t66 = jnp.where(t61 <= p_a_zeta_threshold, t25, t64 * t61)
t69 = r1 ** 2
t70 = r1 ** (0.1e1 / 0.3e1)
t71 = t70 ** 2
t86 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t66 * t31 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t34 * t38 * s2 / t71 / t69 / 0.24e2))))
res = t57 + t86
