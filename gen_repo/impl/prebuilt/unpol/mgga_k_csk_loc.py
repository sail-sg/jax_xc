t3 = 3 ** (0.1e1 / 0.3e1)
t4 = t3 ** 2
t5 = math.pi ** (0.1e1 / 0.3e1)
t8 = 0.1e1 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t11 = lax_cond(t8, -t9, 0)
t12 = lax_cond(t8, t9, t11)
t13 = 0.1e1 + t12
t15 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t15 ** 2
t18 = t13 ** (0.1e1 / 0.3e1)
t19 = t18 ** 2
t21 = lax_cond(t13 <= p_a_zeta_threshold, t16 * p_a_zeta_threshold, t19 * t13)
t22 = r0 ** (0.1e1 / 0.3e1)
t23 = t22 ** 2
t25 = 6 ** (0.1e1 / 0.3e1)
t26 = math.pi ** 2
t27 = t26 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t29 = 0.1e1 / t28
t31 = 2 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = r0 ** 2
t37 = s0 * t32 / t23 / t34
t39 = 0.5e1 / 0.72e2 * t25 * t29 * t37
t52 = params_a_csk_cp * t25 * t29 * t37 / 0.24e2 + params_a_csk_cq * t25 * t29 * l0 * t32 / t23 / r0 / 0.24e2 - t39
t54 = math.log(0.1e1 - DBL_EPSILON)
t55 = 0.1e1 / params_a_csk_a
t56 = (-t54) ** (-t55)
t58 = math.log(DBL_EPSILON)
t59 = (-t58) ** (-t55)
t60 = -t59 < t52
t61 = lax_cond(t60, -t59, t52)
t63 = lax_cond(-t56 < t61, t61, -t56)
t64 = abs(t63)
t65 = t64 ** params_a_csk_a
t67 = math.exp(-0.1e1 / t65)
t69 = (0.1e1 - t67) ** t55
t70 = lax_cond(t60, 1, t69)
t71 = lax_cond(t52 < -t56, 0, t70)
t77 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t4 * t5 * math.pi * t21 * t23 * (t52 * t71 + t39 + 0.1e1))
res = 0.2e1 * t77
