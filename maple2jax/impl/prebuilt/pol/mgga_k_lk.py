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
t33 = 6 ** (0.1e1 / 0.3e1)
t34 = math.pi ** 2
t35 = t34 ** (0.1e1 / 0.3e1)
t36 = t35 ** 2
t38 = t33 / t36
t39 = r0 ** 2
t40 = r0 ** (0.1e1 / 0.3e1)
t41 = t40 ** 2
t43 = 0.1e1 / t41 / t39
t47 = t33 ** 2
t50 = t47 / t35 / t34
t51 = l0 ** 2
t57 = t50 * t51 / t40 / t39 / r0 / 0.5832e4
t58 = t39 ** 2
t64 = t50 * s0 / t40 / t58 * l0 / 0.5184e4
t65 = s0 ** 2
t69 = t65 / t40 / t58 / r0
t71 = t50 * t69 / 0.17496e5
t72 = 0.1e1 / params_a_kappa
t86 = t34 ** 2
t87 = 0.1e1 / t86
t90 = t58 ** 2
t92 = params_a_kappa ** 2
t93 = 0.1e1 / t92
t107 = jnp.where(r0 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t29 * t31 * (0.1e1 + params_a_kappa * (0.2e1 - 0.1e1 / (0.1e1 + (0.5e1 / 0.648e3 * t38 * s0 * t43 + t57 - t64 + t71 + 0.25e2 / 0.419904e6 * t50 * t69 * t72) * t72) - 0.1e1 / (0.1e1 + (0.5e1 / 0.324e3 * t38 * s0 * t43 * (t57 - t64 + t71) * t72 + 0.125e3 / 0.45349632e8 * t87 * t65 * s0 / t90 * t93) * t72))))
t109 = jnp.where(t11, t16, -t18)
t110 = jnp.where(t15, t12, t109)
t111 = 0.1e1 + t110
t113 = t111 ** (0.1e1 / 0.3e1)
t114 = t113 ** 2
t116 = jnp.where(t111 <= p_a_zeta_threshold, t25, t114 * t111)
t118 = r1 ** 2
t119 = r1 ** (0.1e1 / 0.3e1)
t120 = t119 ** 2
t122 = 0.1e1 / t120 / t118
t126 = l1 ** 2
t132 = t50 * t126 / t119 / t118 / r1 / 0.5832e4
t133 = t118 ** 2
t139 = t50 * s2 / t119 / t133 * l1 / 0.5184e4
t140 = s2 ** 2
t144 = t140 / t119 / t133 / r1
t146 = t50 * t144 / 0.17496e5
t162 = t133 ** 2
t177 = jnp.where(r1 <= p_a_dens_threshold, 0, 0.3e1 / 0.20e2 * t6 * t116 * t31 * (0.1e1 + params_a_kappa * (0.2e1 - 0.1e1 / (0.1e1 + (0.5e1 / 0.648e3 * t38 * s2 * t122 + t132 - t139 + t146 + 0.25e2 / 0.419904e6 * t50 * t144 * t72) * t72) - 0.1e1 / (0.1e1 + (0.5e1 / 0.324e3 * t38 * s2 * t122 * (t132 - t139 + t146) * t72 + 0.125e3 / 0.45349632e8 * t87 * t140 * s2 / t162 * t93) * t72))))
res = t107 + t177
