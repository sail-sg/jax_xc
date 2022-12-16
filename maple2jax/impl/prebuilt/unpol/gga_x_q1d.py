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
t25 = 0.1e1 / t24
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t34 = s0 * t28 / t31 / t30
t35 = t21 * t25 * t34
t39 = 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t35)
t41 = t21 ** 2
t45 = s0 ** 2
t47 = t30 ** 2
t53 = t41 / t23 / t22 * t45 * t27 / t19 / t47 / r0 / 0.288e3
t55 = t22 ** 2
t59 = t47 ** 2
t77 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.18040e1 - t39 + (t35 / 0.24e2 + t53) / (0.1e1 + t53 + 0.1e1 / t55 * t45 * s0 / t59 / 0.576e3) * (-(0.18040e1 - t39) * t21 * t25 * t34 / 0.24e2 + 0.6525e-1)))
res = 0.2e1 * t77
