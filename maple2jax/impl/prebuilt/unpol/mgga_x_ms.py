t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t34 = s0 * t28 / t31 / t30
t36 = 0.5e1 / 0.972e3 * t21 / t24 * t34
t41 = params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t36))
t47 = tau0 * t28 / t31 / r0 - t34 / 0.8e1
t48 = t47 ** 2
t49 = t21 ** 2
t55 = 0.1e1 - 0.25e2 / 0.81e2 * t48 * t49 / t23 / t22
t56 = t55 ** 2
t59 = t22 ** 2
t63 = t48 ** 2
t66 = t59 ** 2
t84 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + t41 + t56 * t55 / (0.1e1 + 0.250e3 / 0.243e3 * t48 * t47 / t59 + 0.62500e5 / 0.59049e5 * params_a_b * t63 * t48 / t66) * (params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + t36 + params_a_c)) - t41)))
res = 0.2e1 * t84
