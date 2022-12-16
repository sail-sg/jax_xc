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
t19 = jnp.where(t15, t16, t18)
t20 = jnp.where(t11, t12, t19)
t21 = 0.1e1 + t20
t23 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = t24 * p_a_zeta_threshold
t26 = t21 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t29 = jnp.where(t21 <= p_a_zeta_threshold, t25, t27 * t21)
t30 = t7 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t35 = 6 ** (0.1e1 / 0.3e1)
t36 = (params_a_muPBE - params_a_muGE) * params_a_alpha * t35
t37 = math.pi ** 2
t38 = t37 ** (0.1e1 / 0.3e1)
t39 = t38 ** 2
t40 = 0.1e1 / t39
t41 = t40 * s0
t42 = r0 ** 2
t43 = r0 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t46 = 0.1e1 / t44 / t42
t47 = params_a_alpha * t35
t48 = t41 * t46
t70 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + (params_a_muGE + t36 * t41 * t46 / (0.1e1 + t47 * t48 / 0.24e2) / 0.24e2) * t35 * t48 / 0.24e2))))
t72 = jnp.where(t11, t16, -t18)
t73 = jnp.where(t15, t12, t72)
t74 = 0.1e1 + t73
t76 = t74 ** (0.1e1 / 0.3e1)
t77 = t76 ** 2
t79 = jnp.where(t74 <= p_a_zeta_threshold, t25, t77 * t74)
t81 = t40 * s2
t82 = r1 ** 2
t83 = r1 ** (0.1e1 / 0.3e1)
t84 = t83 ** 2
t86 = 0.1e1 / t84 / t82
t87 = t81 * t86
t109 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t79 * t31 * (0.1e1 + params_a_kappa * (0.1e1 - params_a_kappa / (params_a_kappa + (params_a_muGE + t36 * t81 * t86 / (0.1e1 + t47 * t87 / 0.24e2) / 0.24e2) * t35 * t87 / 0.24e2))))
res = t70 + t109
