t2 = 3 ** (0.1e1 / 0.3e1)
t3 = math.pi ** (0.1e1 / 0.3e1)
t5 = t2 / t3
t6 = r0 + r1
t7 = 0.1e1 / t6
t10 = 0.2e1 * r0 * t7 <= p_a_zeta_threshold
t11 = p_a_zeta_threshold - 0.1e1
t14 = 0.2e1 * r1 * t7 <= p_a_zeta_threshold
t15 = -t11
t17 = (r0 - r1) * t7
t18 = lax_cond(t14, t15, t17)
t19 = lax_cond(t10, t11, t18)
t20 = 0.1e1 + t19
t22 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t23 = t22 * p_a_zeta_threshold
t24 = t20 ** (0.1e1 / 0.3e1)
t26 = lax_cond(t20 <= p_a_zeta_threshold, t23, t24 * t20)
t27 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t40 = s0 / t37 / t35
t47 = (t34 * t40 / (0.91464571985215458336e0 * t34 * t40 + 0.8040e0)) ** 0.1000e3
t57 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1e1 - 0.91464571985215458336e-2 * t34 * t40 * (0.13344141567995010044e-3 * t47 - 0.1e1)))
t59 = lax_cond(t10, t15, -t17)
t60 = lax_cond(t14, t11, t59)
t61 = 0.1e1 + t60
t63 = t61 ** (0.1e1 / 0.3e1)
t65 = lax_cond(t61 <= p_a_zeta_threshold, t23, t63 * t61)
t67 = r1 ** 2
t68 = r1 ** (0.1e1 / 0.3e1)
t69 = t68 ** 2
t72 = s2 / t69 / t67
t79 = (t34 * t72 / (0.91464571985215458336e0 * t34 * t72 + 0.8040e0)) ** 0.1000e3
t89 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t65 * t27 * (0.1e1 - 0.91464571985215458336e-2 * t34 * t72 * (0.13344141567995010044e-3 * t79 - 0.1e1)))
res = t57 + t89
