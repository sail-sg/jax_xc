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
t22 = params_a_alpha * t21
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t26 = 0.1e1 / t25
t27 = t22 * t26
t28 = 2 ** (0.1e1 / 0.3e1)
t29 = t28 ** 2
t30 = s0 * t29
t31 = r0 ** 2
t32 = t19 ** 2
t34 = 0.1e1 / t32 / t31
t35 = t30 * t34
t39 = 0.1e1 / (0.1e1 + t27 * t35 / 0.24e2)
t53 = t21 ** 2
t55 = 0.1e1 / t24
t56 = math.sqrt(s0)
t61 = 0.1e1 / t19 / r0
t64 = t3 ** 2
t65 = math.sqrt(0.12e2)
t69 = t53 * t55 * t56 * t28 * t61
t70 = math.sqrt(t69)
t73 = math.sqrt(0.6e1)
t76 = scipy.special.lambertw(t65 * t70 * t69 * t73 / 0.1728e4)
t77 = t76 ** (0.1e1 / 0.3e1)
t78 = t77 ** 2
t85 = (0.2823705740248932030511071641312341561894e2 + 0.3e1 / 0.4e1 * t3 * t29 * t77 * t76) ** (0.1e1 / 0.4e1)
t100 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.1e1 - t27 * t30 * t34 * t39 / 0.24e2 + t22 * t26 * s0 * t29 * t34 * t39 * (0.1e1 + params_a_c * t21 * t26 * t35 / 0.24e2) / (0.1e1 + params_a_c * t53 * t55 * t56 * t29 * t61 / math.pi * t64 * t78 * t85 / 0.8e1) / 0.24e2))
res = 0.2e1 * t100
