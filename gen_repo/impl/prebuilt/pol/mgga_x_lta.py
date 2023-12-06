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
t29 = r0 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t34 = 6 ** (0.1e1 / 0.3e1)
t35 = math.pi ** 2
t36 = t35 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t39 = t34 / t37
t42 = 0.4e1 / 0.5e1 * params_a_ltafrac
t43 = (0.5e1 / 0.9e1 * tau0 / t30 / r0 * t39) ** t42
t47 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * t43)
t49 = lax_cond(t10, t15, -t17)
t50 = lax_cond(t14, t11, t49)
t51 = 0.1e1 + t50
t53 = t51 ** (0.1e1 / 0.3e1)
t55 = lax_cond(t51 <= p_a_zeta_threshold, t23, t53 * t51)
t57 = r1 ** (0.1e1 / 0.3e1)
t58 = t57 ** 2
t64 = (0.5e1 / 0.9e1 * tau1 / t58 / r1 * t39) ** t42
t68 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t55 * t27 * t64)
res = t47 + t68
