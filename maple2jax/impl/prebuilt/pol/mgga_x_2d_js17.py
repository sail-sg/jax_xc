t2 = math.sqrt(math.pi)
t3 = 0.1e1 / t2
t4 = r0 + r1
t5 = 0.1e1 / t4
t8 = 0.2e1 * r0 * t5 <= p_a_zeta_threshold
t9 = p_a_zeta_threshold - 0.1e1
t12 = 0.2e1 * r1 * t5 <= p_a_zeta_threshold
t13 = -t9
t15 = (r0 - r1) * t5
t16 = lax_cond(t12, t13, t15)
t17 = lax_cond(t8, t9, t16)
t18 = 0.1e1 + t17
t20 = math.sqrt(p_a_zeta_threshold)
t21 = t20 * p_a_zeta_threshold
t22 = math.sqrt(t18)
t24 = lax_cond(t18 <= p_a_zeta_threshold, t21, t22 * t18)
t26 = math.sqrt(0.2e1)
t27 = math.sqrt(t4)
t28 = t26 * t27
t29 = 0.1e1 / math.pi
t31 = r0 ** 2
t34 = t29 * s0 / t31 / r0
t36 = math.pi ** 2
t37 = 0.1e1 / t36
t38 = s0 ** 2
t40 = t31 ** 2
t45 = 0.1e1 + 0.12960000000000000000e1 * t34 + 0.62208000000000000000e-2 * t37 * t38 / t40 / t31
t46 = t45 ** (0.1e1 / 0.15e2)
t52 = 0.36912000000000000000e1 * math.pi
t57 = t45 ** (0.1e1 / 0.5e1)
t65 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t24 * t28 * (0.1e1 / t46 + 0.2e1 / 0.5e1 * (0.1e1 + 0.87771428571428571429e-1 * t34 + (-0.7720000000000000000e-1 * tau0 / t31 - t52) * t29 / 0.4e1) / t57))
t67 = lax_cond(t8, t13, -t15)
t68 = lax_cond(t12, t9, t67)
t69 = 0.1e1 + t68
t71 = math.sqrt(t69)
t73 = lax_cond(t69 <= p_a_zeta_threshold, t21, t71 * t69)
t76 = r1 ** 2
t79 = t29 * s2 / t76 / r1
t81 = s2 ** 2
t83 = t76 ** 2
t88 = 0.1e1 + 0.12960000000000000000e1 * t79 + 0.62208000000000000000e-2 * t37 * t81 / t83 / t76
t89 = t88 ** (0.1e1 / 0.15e2)
t99 = t88 ** (0.1e1 / 0.5e1)
t107 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.2e1 / 0.3e1 * t3 * t73 * t28 * (0.1e1 / t89 + 0.2e1 / 0.5e1 * (0.1e1 + 0.87771428571428571429e-1 * t79 + (-0.7720000000000000000e-1 * tau1 / t76 - t52) * t29 / 0.4e1) / t99))
res = t65 + t107
