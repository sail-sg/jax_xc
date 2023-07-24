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
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = 2 ** (0.1e1 / 0.3e1)
t22 = t3 ** 2
t24 = 4 ** (0.1e1 / 0.3e1)
t26 = 0.8e1 / 0.27e2 * t21 * t22 * t24
t27 = t21 ** 2
t29 = t20 ** 2
t34 = r0 ** 2
t37 = s0 * t27 / t29 / t34
t40 = 6 ** (0.1e1 / 0.3e1)
t42 = math.pi ** 2
t43 = t42 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t46 = (tau0 * t27 / t29 / r0 - t37 / 0.8e1) * t40 / t44
t58 = 0.5e1 / 0.9e1 * t46
t59 = 0.1e1 - t58
t60 = Heaviside(t59)
t68 = Heaviside(-t59)
t78 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((t26 + 0.5e1 / 0.9e1 * t46 * (0.827411e0 - 0.35753333333333333333e0 * t46) / (0.10e1 - 0.45341611111111111111e0 * t46) * (0.1e1 - t26)) * t60 + (0.1e1 + 0.148e0 * t59 / (0.1e1 + t58)) * t68) / (0.1e1 + 0.1015549e-2 * t37))
res = 0.2e1 * t78
