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
t31 = r0 ** 2
t32 = t19 ** 2
t35 = s0 * t29 / t32 / t31
t38 = math.exp(-params_a_alpha * t21 * t26 * t35 / 0.24e2)
t45 = t21 ** 2
t46 = 0.1e1 / t24
t47 = t45 * t46
t48 = math.sqrt(s0)
t51 = 0.1e1 / t19 / r0
t52 = t48 * t28 * t51
t55 = (t47 * t52 / 0.12e2) ** params_a_expo
t56 = params_a_f * t55
t64 = math.asinh(params_a_b * t45 * t46 * t52 / 0.12e2)
t76 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 + ((params_a_d * t38 + params_a_c) * t21 * t26 * t35 / 0.24e2 - t56) / (0.1e1 + t47 * t48 * t28 * t51 * params_a_a * t64 / 0.12e2 + t56)))
res = 0.2e1 * t76
