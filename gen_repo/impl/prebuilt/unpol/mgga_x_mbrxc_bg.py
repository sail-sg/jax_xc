t3 = 0.1e1 <= p_a_zeta_threshold
t4 = p_a_zeta_threshold - 0.1e1
t6 = lax_cond(t3, -t4, 0)
t7 = lax_cond(t3, t4, t6)
t8 = 0.1e1 + t7
t10 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t12 = t8 ** (0.1e1 / 0.3e1)
t14 = lax_cond(t8 <= p_a_zeta_threshold, t10 * p_a_zeta_threshold, t12 * t8)
t15 = r0 ** (0.1e1 / 0.3e1)
t17 = 32 ** (0.1e1 / 0.3e1)
t19 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t23 = 4 ** (0.1e1 / 0.3e1)
t24 = 2 ** (0.1e1 / 0.3e1)
t25 = t24 ** 2
t27 = t15 ** 2
t32 = 6 ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t34 = math.pi ** 2
t35 = t34 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t40 = r0 ** 2
t45 = s0 ** 2
t47 = t40 ** 2
t53 = 0.149492e0 * tau0 * t25 / t27 / r0 - 0.3e1 / 0.10e2 * t33 * t36 + 0.147e0 * s0 * t25 / t27 / t40 + 0.64e-2 * t45 * t24 / t15 / t47 / r0
t54 = abs(t53)
t57 = lax_cond(0.0e0 < t53, 0.50e-12, -0.50e-12)
t58 = lax_cond(t54 < 0.50e-12, t57, t53)
t59 = mbrxc_x(t58)
t61 = math.exp(t59 / 0.3e1)
t63 = math.exp(-t59)
t64 = t59 ** 2
t72 = (0.1e1 + t59) ** (0.1e1 / 0.3e1)
t78 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -t14 * t15 * t17 / t19 * t23 * t61 * (0.8e1 - t63 * (t64 + 0.5e1 * t59 + 0.8e1)) / t59 / t72 / 0.64e2)
res = 0.2e1 * t78
