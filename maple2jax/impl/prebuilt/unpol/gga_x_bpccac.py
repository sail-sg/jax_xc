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
t21 = math.sqrt(s0)
t22 = 2 ** (0.1e1 / 0.3e1)
t25 = 0.1e1 / t19 / r0
t26 = t21 * t22 * t25
t28 = math.exp(-t26 + 0.19e2)
t30 = 0.1e1 / (0.1e1 + t28)
t32 = 6 ** (0.1e1 / 0.3e1)
t33 = math.pi ** 2
t34 = t33 ** (0.1e1 / 0.3e1)
t35 = t34 ** 2
t36 = 0.1e1 / t35
t38 = t22 ** 2
t40 = r0 ** 2
t41 = t19 ** 2
t44 = s0 * t38 / t41 / t40
t45 = t32 * t36 * t44
t53 = math.exp(-0.25e2 / 0.6e1 * t45)
t60 = t32 ** 2
t64 = s0 ** 2
t66 = t40 ** 2
t72 = 0.13888888888888888889e-4 * t60 / t34 / t33 * t64 * t22 / t19 / t66 / r0
t75 = t60 / t34
t80 = math.asinh(0.64963333333333333333e0 * t75 * t26)
t93 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * ((0.1e1 - t30) * (0.2227e1 - 0.1505529e1 / (0.1227e1 + 0.91464571985215458336e-2 * t45)) + t30 * (0.1e1 + ((0.2743e0 - 0.1508e0 * t53) * t32 * t36 * t44 / 0.24e2 - t72) / (0.1e1 + 0.16370833333333333333e-1 * t75 * t21 * t22 * t25 * t80 + t72))))
res = 0.2e1 * t93
