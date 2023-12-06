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
t33 = 0.1e1 / t32
t34 = t29 * t33
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t39 = 0.1e1 / t37 / t35
t41 = t34 * s0 * t39
t49 = t34 * tau0 / t37 / r0 / 0.4e1 - 0.9e1 / 0.20e2 - t41 / 0.288e3
t50 = t49 ** 2
t57 = t29 ** 2
t60 = t57 / t31 / t30
t61 = s0 ** 2
t62 = t35 ** 2
t76 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.1804e1 - 0.646416e0 / (0.804e0 + 0.5e1 / 0.972e3 * t41 + 0.146e3 / 0.2025e4 * t50 - 0.73e2 / 0.9720e4 * t49 * t29 * t33 * s0 * t39 + 0.22909234000912809658e-3 * t60 * t61 / t36 / t62 / r0)))
t78 = lax_cond(t10, t15, -t17)
t79 = lax_cond(t14, t11, t78)
t80 = 0.1e1 + t79
t82 = t80 ** (0.1e1 / 0.3e1)
t84 = lax_cond(t80 <= p_a_zeta_threshold, t23, t82 * t80)
t86 = r1 ** 2
t87 = r1 ** (0.1e1 / 0.3e1)
t88 = t87 ** 2
t90 = 0.1e1 / t88 / t86
t92 = t34 * s2 * t90
t100 = t34 * tau1 / t88 / r1 / 0.4e1 - 0.9e1 / 0.20e2 - t92 / 0.288e3
t101 = t100 ** 2
t108 = s2 ** 2
t109 = t86 ** 2
t123 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t84 * t27 * (0.1804e1 - 0.646416e0 / (0.804e0 + 0.5e1 / 0.972e3 * t92 + 0.146e3 / 0.2025e4 * t101 - 0.73e2 / 0.9720e4 * t100 * t29 * t33 * s2 * t90 + 0.22909234000912809658e-3 * t60 * t108 / t87 / t109 / r1)))
res = t76 + t123
