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
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t26 = 0.1e1 / t25
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = s0 * t29
t31 = r0 ** 2
t32 = t19 ** 2
t34 = 0.1e1 / t32 / t31
t40 = 0.1e1 + params_a_a * t21 * t26 * t30 * t34 / 0.24e2
t46 = t21 ** 2
t51 = s0 ** 2
t53 = t31 ** 2
t57 = t40 ** 2
t63 = t23 ** 2
t67 = t53 ** 2
t79 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + params_a_c1 * t21 * t26 * t30 * t34 / t40 / 0.24e2 + params_a_c2 * t46 / t24 / t23 * t51 * t28 / t19 / t53 / r0 / t57 / 0.288e3 + params_a_c3 / t63 * t51 * s0 / t67 / t57 / t40 / 0.576e3))
res = 0.2e1 * t79
