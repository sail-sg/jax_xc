t1 = r0 + r1
t2 = r0 - r1
t4 = t2 / t1
t5 = abs(t4)
t8 = t2 ** 2
t9 = t1 ** 2
t13 = t1 ** (0.1e1 / 0.3e1)
t17 = p_a_zeta_threshold - 0.1e1
t21 = lax_cond(0.1e1 - t4 <= p_a_zeta_threshold, -t17, t4)
t22 = lax_cond(0.1e1 + t4 <= p_a_zeta_threshold, t17, t21)
t23 = 0.1e1 + t22
t24 = t23 ** params_a_q
t25 = 0.1e1 - t22
t26 = t25 ** params_a_q
t27 = t24 + t26
t28 = t22 ** 2
t30 = (0.1e1 - t28) ** (0.1e1 / 0.3e1)
t32 = t23 ** (0.1e1 / 0.3e1)
t33 = t25 ** (0.1e1 / 0.3e1)
t34 = t32 + t33
t42 = 0.1e1 / t13
t43 = 0.1e1 / params_a_fc
t48 = 0.1e1 / t27 / t30 * t34
t49 = t42 * t43 * t48
t52 = math.log(0.1e1 + 0.91959623973811018799e-1 * t49)
t58 = t13 ** 2
t60 = params_a_fc ** 2
t63 = t27 ** 2
t65 = t30 ** 2
t68 = t34 ** 2
t75 = lax_cond(0.1e1 - t5 <= p_a_zeta_threshold, 0, (0.1e1 - t8 / t9) * (-0.2763169e1 / (0.1e1 + 0.10874334072525e2 * t13 * params_a_fc * t27 * t30 / t34) + 0.28144540420067767464e0 * t52 * t42 * t43 * t48 + 0.25410002852601321894e0 * t49 - 0.49248579417833934399e-1 / t58 / t60 / t63 / t65 * t68) / 0.4e1)
res = t1 * t75
