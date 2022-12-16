t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = jnp.where(t7, -t8, 0)
t11 = jnp.where(t7, t8, t10)
t12 = 0.1e1 + t11
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = jnp.where(t12 <= p_a_zeta_threshold, t14 * p_a_zeta_threshold, t16 * t12)
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = 6 ** (0.1e1 / 0.3e1)
t22 = math.pi ** 2
t23 = t22 ** (0.1e1 / 0.3e1)
t24 = t23 ** 2
t25 = 0.1e1 / t24
t27 = 2 ** (0.1e1 / 0.3e1)
t28 = t27 ** 2
t30 = r0 ** 2
t31 = t20 ** 2
t33 = 0.1e1 / t31 / t30
t34 = s0 * t28 * t33
t39 = 0.100e3 / 0.6561e4 / params_a_k1 - 0.73e2 / 0.648e3
t40 = t21 ** 2
t45 = s0 ** 2
t47 = t30 ** 2
t55 = math.exp(-0.27e2 / 0.80e2 * t39 * t21 * t25 * t34)
t60 = math.sqrt(0.146e3)
t79 = (tau0 * t28 / t31 / r0 - t34 / 0.8e1) / (0.3e1 / 0.10e2 * t40 * t24 + params_a_eta * s0 * t28 * t33 / 0.8e1)
t80 = 0.1e1 - t79
t82 = t80 ** 2
t84 = math.exp(-t82 / 0.2e1)
t88 = (0.7e1 / 0.12960e5 * t60 * t21 * t25 * t34 + t60 * t80 * t84 / 0.100e3) ** 2
t96 = 0.25e1 < t79
t97 = jnp.where(t96, 0.25e1, t79)
t99 = t97 ** 2
t101 = t99 * t97
t103 = t99 ** 2
t112 = jnp.where(t96, t79, 0.25e1)
t116 = math.exp(params_a_c2 / (0.1e1 - t112))
t118 = jnp.where(t79 <= 0.25e1, 0.1e1 - 0.667e0 * t97 - 0.4445555e0 * t99 - 0.663086601049e0 * t101 + 0.1451297044490e1 * t103 - 0.887998041597e0 * t103 * t97 + 0.234528941479e0 * t103 * t99 - 0.23185843322e-1 * t103 * t101, -params_a_d * t116)
t124 = math.sqrt(0.3e1)
t127 = math.sqrt(s0)
t133 = math.sqrt(t40 / t23 * t127 * t27 / t20 / r0)
t137 = math.exp(-0.98958000000000000000e1 * t124 / t133)
t142 = jnp.where(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t20 * ((0.1e1 + params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + 0.5e1 / 0.972e3 * t21 * t25 * t34 + t39 * t40 / t23 / t22 * t45 * t27 / t20 / t47 / r0 * t55 / 0.288e3 + t88))) * (0.1e1 - t118) + 0.1174e1 * t118) * (0.1e1 - t137))
res = 0.2e1 * t142
