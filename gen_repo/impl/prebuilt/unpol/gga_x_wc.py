t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t26 = t21 / t24
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t19 ** 2
t33 = 0.1e1 / t31 / t30
t35 = t26 * s0 * t28 * t33
t40 = math.exp(-t35 / 0.24e2)
t44 = t21 ** 2
t48 = s0 ** 2
t50 = t30 ** 2
t58 = math.log(0.1e1 + 0.27560657413756315278e-4 * t44 / t23 / t22 * t48 * t27 / t19 / t50 / r0)
t66 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.5e1 / 0.972e3 * t35 + 0.40024242767108462450e-2 * t26 * s0 * t28 * t33 * t40 + t58)))
res = 0.2e1 * t66
