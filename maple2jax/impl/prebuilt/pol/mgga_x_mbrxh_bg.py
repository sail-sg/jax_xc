t2 = r0 + r1
t3 = 0.1e1 / t2
t6 = 0.2e1 * r0 * t3 <= p_a_zeta_threshold
t7 = p_a_zeta_threshold - 0.1e1
t10 = 0.2e1 * r1 * t3 <= p_a_zeta_threshold
t11 = -t7
t13 = (r0 - r1) * t3
t14 = jnp.where(t10, t11, t13)
t15 = jnp.where(t6, t7, t14)
t16 = 0.1e1 + t15
t18 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t19 = t18 * p_a_zeta_threshold
t20 = t16 ** (0.1e1 / 0.3e1)
t22 = jnp.where(t16 <= p_a_zeta_threshold, t19, t20 * t16)
t23 = t2 ** (0.1e1 / 0.3e1)
t26 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t27 = 0.1e1 / t26
t29 = 4 ** (0.1e1 / 0.3e1)
t30 = r0 ** (0.1e1 / 0.3e1)
t31 = t30 ** 2
t36 = 6 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t38 = math.pi ** 2
t39 = t38 ** (0.1e1 / 0.3e1)
t40 = t39 ** 2
t42 = 0.3e1 / 0.10e2 * t37 * t40
t43 = r0 ** 2
t48 = s0 ** 2
t49 = t43 ** 2
t55 = 0.46864e0 * tau0 / t31 / r0 - t42 + 0.89e-1 * s0 / t31 / t43 + 0.53e-2 * t48 / t30 / t49 / r0
t56 = abs(t55)
t59 = jnp.where(0.0e0 < t55, 0.50e-12, -0.50e-12)
t60 = jnp.where(t56 < 0.50e-12, t59, t55)
t61 = br89_x(t60)
t63 = math.exp(t61 / 0.3e1)
t65 = math.exp(-t61)
t75 = jnp.where(r0 <= p_a_dens_threshold, 0, -t22 * t23 * t27 * t29 * t63 * (0.1e1 - t65 * (0.1e1 + t61 / 0.2e1)) / t61 / 0.4e1)
t77 = jnp.where(t6, t11, -t13)
t78 = jnp.where(t10, t7, t77)
t79 = 0.1e1 + t78
t81 = t79 ** (0.1e1 / 0.3e1)
t83 = jnp.where(t79 <= p_a_zeta_threshold, t19, t81 * t79)
t86 = r1 ** (0.1e1 / 0.3e1)
t87 = t86 ** 2
t92 = r1 ** 2
t97 = s2 ** 2
t98 = t92 ** 2
t104 = 0.46864e0 * tau1 / t87 / r1 - t42 + 0.89e-1 * s2 / t87 / t92 + 0.53e-2 * t97 / t86 / t98 / r1
t105 = abs(t104)
t108 = jnp.where(0.0e0 < t104, 0.50e-12, -0.50e-12)
t109 = jnp.where(t105 < 0.50e-12, t108, t104)
t110 = br89_x(t109)
t112 = math.exp(t110 / 0.3e1)
t114 = math.exp(-t110)
t124 = jnp.where(r1 <= p_a_dens_threshold, 0, -t83 * t23 * t27 * t29 * t112 * (0.1e1 - t114 * (0.1e1 + t110 / 0.2e1)) / t110 / 0.4e1)
res = t75 + t124
