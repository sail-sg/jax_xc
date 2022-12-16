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
t23 = 6 ** (0.1e1 / 0.3e1)
t25 = math.pi ** 2
t26 = t25 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t32 = 2 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = r0 ** 2
t35 = t19 ** 2
t41 = params_a_mu * t23 / t27 / params_a_kappa * s0 * t33 / t35 / t34 / 0.24e2
t44 = 0.1e1 - 0.1e1 / (0.1e1 + t41)
t47 = t44 ** 2
t53 = t47 ** 2
t60 = math.exp(-t41)
t61 = 0.1e1 - t60
t64 = t61 ** 2
t70 = t64 ** 2
t75 = params_a_a[3] * t47 * t44 + params_a_a[5] * t53 * t44 + params_a_b[3] * t64 * t61 + params_a_b[5] * t70 * t61 + params_a_a[1] * t44 + params_a_a[2] * t47 + params_a_a[4] * t53 + params_a_b[1] * t61 + params_a_b[2] * t64 + params_a_b[4] * t70 + params_a_a[0] + params_a_b[0]
t79 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * t75)
res = 0.2e1 * t79
