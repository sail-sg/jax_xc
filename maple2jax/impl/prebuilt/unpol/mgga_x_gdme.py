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
t23 = 2 ** (0.1e1 / 0.3e1)
t26 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t27 = 0.1e1 / t26
t28 = 4 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t39 = t23 ** 2
t43 = params_a_a ** 2
t46 = t19 ** 2
t48 = 0.1e1 / t46 / r0
t63 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.2e1 / 0.9e1 * (params_a_AA + 0.3e1 / 0.5e1 * params_a_BB) * t23 * t27 * t28 / t32 + params_a_BB * t3 * t27 * t28 * t39 / t31 / t30 * ((t43 - params_a_a + 0.1e1 / 0.2e1) * l0 * t39 * t48 - 0.2e1 * tau0 * t39 * t48) / 0.27e2))
res = 0.2e1 * t63
