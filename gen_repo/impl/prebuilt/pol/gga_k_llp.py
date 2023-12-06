t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t6 = t3 * t4 * math.pi
t7 = r0 + r1
t8 = 0.1e1 / t7
t11 = 0.2e1 * r0 * t8 <= p_a_zeta_threshold
t12 = p_a_zeta_threshold - 0.1e1
t15 = 0.2e1 * r1 * t8 <= p_a_zeta_threshold
t16 = -t12
t18 = (r0 - r1) * t8
t19 = lax_cond(t15, t16, t18)
t20 = lax_cond(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = lax_cond(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t35 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t37 = params_a_beta * t3 / t35
t38 = 4 ** (0.1e1 / 0.3e1)
t40 = r0 ** 2
t41 = r0 ** (0.1e1 / 0.3e1)
t42 = t41 ** 2
t45 = params_a_gamma * params_a_beta
t46 = math.sqrt(s0)
t49 = t46 / t41 / r0
t50 = math.asinh(t49)
t63 = lax_cond(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.10e1 + 0.2e1 / 0.9e1 * t37 * t38 * s0 / t42 / t40 / (0.10e1 + t45 * t49 * t50)))
t65 = lax_cond(t11, t16, -t18)
t66 = lax_cond(t15, t12, t65)
t67 = 0.1e1 + t66
t69 = t67 ** (0.1e1 / 0.3e1)
t70 = t69 ** 2
t72 = lax_cond(t67 <= p_a_zeta_threshold, t25, t70 * t67)
t75 = r1 ** 2
t76 = r1 ** (0.1e1 / 0.3e1)
t77 = t76 ** 2
t80 = math.sqrt(s2)
t83 = t80 / t76 / r1
t84 = math.asinh(t83)
t97 = lax_cond(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t72 * t31 * (0.10e1 + 0.2e1 / 0.9e1 * t37 * t38 * s2 / t77 / t75 / (0.10e1 + t45 * t83 * t84)))
res = t63 + t97
