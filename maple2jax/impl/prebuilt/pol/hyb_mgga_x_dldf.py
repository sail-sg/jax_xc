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
t28 = t6 ** (0.1e1 / 0.3e1)
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t48 = t29 ** 2
t50 = 0.3e1 / 0.10e2 * t48 * t32
t53 = tau0 / t37 / r0
t54 = t50 - t53
t55 = t50 + t53
t59 = t54 ** 2
t60 = t55 ** 2
t69 = t59 ** 2
t70 = t60 ** 2
t78 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.14459516250000000000e0 * t5 * t26 * t28 * (0.58827323e1 - 0.2384107471346329e2 / (0.48827323e1 + 0.14629700000000000000e-1 * t34 * s0 / t37 / t35)) * (0.1e1 - 0.1637571e0 * t54 / t55 - 0.1880028e0 * t59 / t60 - 0.4490609e0 * t59 * t54 / t60 / t55 - 0.82359e-2 * t69 / t70))
t80 = lax_cond(t10, t15, -t17)
t81 = lax_cond(t14, t11, t80)
t82 = 0.1e1 + t81
t84 = t82 ** (0.1e1 / 0.3e1)
t86 = lax_cond(t82 <= p_a_zeta_threshold, t23, t84 * t82)
t88 = r1 ** 2
t89 = r1 ** (0.1e1 / 0.3e1)
t90 = t89 ** 2
t103 = tau1 / t90 / r1
t104 = t50 - t103
t105 = t50 + t103
t109 = t104 ** 2
t110 = t105 ** 2
t119 = t109 ** 2
t120 = t110 ** 2
t128 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.14459516250000000000e0 * t5 * t86 * t28 * (0.58827323e1 - 0.2384107471346329e2 / (0.48827323e1 + 0.14629700000000000000e-1 * t34 * s2 / t90 / t88)) * (0.1e1 - 0.1637571e0 * t104 / t105 - 0.1880028e0 * t109 / t110 - 0.4490609e0 * t109 * t104 / t110 / t105 - 0.82359e-2 * t119 / t120))
res = t78 + t128
