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
t22 = t21 ** 2
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = 0.1e1 / t24
t27 = math.sqrt(s0)
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t27 * t28
t31 = 0.1e1 / t19 / r0
t32 = t29 * t31
t34 = t22 * t25 * t32 / 0.12e2
t35 = math.tanh(t34)
t37 = math.asinh(t34)
t42 = math.log(0.1e1 + t34)
t64 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params_a_mu * t35 * t37 * (0.1e1 + params_a_alpha * ((0.1e1 - params_a_zeta) * t22 * t25 * t29 * t31 * t42 / 0.12e2 + params_a_zeta * t22 * t25 * t32 / 0.12e2)) / (params_a_beta * t35 * t37 + 0.1e1)))
res = 0.2e1 * t64
