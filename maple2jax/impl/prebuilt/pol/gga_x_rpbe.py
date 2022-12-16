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
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = params_a_rpbe_mu * t29 / t33
t36 = r0 ** 2
t37 = r0 ** (0.1e1 / 0.3e1)
t38 = t37 ** 2
t42 = 0.1e1 / params_a_rpbe_kappa
t46 = math.exp(-t35 * s0 / t38 / t36 * t42 / 0.24e2)
t53 = jnp.where(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 + params_a_rpbe_kappa * (0.1e1 - t46)))
t55 = jnp.where(t10, t15, -t17)
t56 = jnp.where(t14, t11, t55)
t57 = 0.1e1 + t56
t59 = t57 ** (0.1e1 / 0.3e1)
t61 = jnp.where(t57 <= p_a_zeta_threshold, t23, t59 * t57)
t63 = r1 ** 2
t64 = r1 ** (0.1e1 / 0.3e1)
t65 = t64 ** 2
t72 = math.exp(-t35 * s2 / t65 / t63 * t42 / 0.24e2)
t79 = jnp.where(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t61 * t27 * (0.1e1 + params_a_rpbe_kappa * (0.1e1 - t72)))
res = t53 + t79
