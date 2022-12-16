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
t21 = 2 ** (0.1e1 / 0.3e1)
t22 = t3 ** 2
t24 = 4 ** (0.1e1 / 0.3e1)
t26 = 0.8e1 / 0.27e2 * t21 * t22 * t24
t27 = t21 ** 2
t29 = t19 ** 2
t34 = r0 ** 2
t39 = tau0 * t27 / t29 / r0 - s0 * t27 / t29 / t34 / 0.8e1
t40 = 6 ** (0.1e1 / 0.3e1)
t42 = math.pi ** 2
t43 = t42 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t45 = 0.1e1 / t44
t46 = t39 * t40 * t45
t48 = t40 * t45
t64 = 0.5e1 / 0.9e1 * t46
t65 = 0.1e1 - t64
t66 = Heaviside(t65)
t75 = Heaviside(-t65)
t81 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((t26 + 0.5e1 / 0.9e1 * t46 * (params_a_c0 + 0.5e1 / 0.9e1 * params_a_c1 * t39 * t48) / (0.10e1 + 0.5e1 / 0.9e1 * (params_a_c0 + params_a_c1 - 0.1e1) * t39 * t48) * (0.1e1 - t26)) * t66 + (0.1e1 + (0.1e1 - params_a_alphainf) * t65 / (0.1e1 + t64)) * t75))
res = 0.2e1 * t81
