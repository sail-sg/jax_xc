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
t33 = 0.1e1 / t32
t34 = t29 * t33
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t39 = 0.1e1 / t37 / t35
t40 = s0 * t39
t45 = 0.100e3 / 0.6561e4 / params_a_k1 - 0.73e2 / 0.648e3
t46 = t29 ** 2
t50 = t45 * t46 / t31 / t30
t51 = s0 ** 2
t52 = t35 ** 2
t57 = t45 * t29
t59 = t33 * s0 * t39
t62 = math.exp(-0.27e2 / 0.80e2 * t57 * t59)
t66 = math.sqrt(0.146e3)
t67 = t66 * t29
t76 = 0.3e1 / 0.10e2 * t46 * t32
t82 = (tau0 / t37 / r0 - t40 / 0.8e1) / (t76 + params_a_eta * s0 * t39 / 0.8e1)
t83 = 0.1e1 - t82
t85 = t83 ** 2
t87 = math.exp(-t85 / 0.2e1)
t91 = (0.7e1 / 0.12960e5 * t67 * t59 + t66 * t83 * t87 / 0.100e3) ** 2
t99 = 0.25e1 < t82
t100 = lax_cond(t99, 0.25e1, t82)
t102 = t100 ** 2
t104 = t102 * t100
t106 = t102 ** 2
t115 = lax_cond(t99, t82, 0.25e1)
t119 = math.exp(params_a_c2 / (0.1e1 - t115))
t121 = lax_cond(t82 <= 0.25e1, 0.1e1 - 0.667e0 * t100 - 0.4445555e0 * t102 - 0.663086601049e0 * t104 + 0.1451297044490e1 * t106 - 0.887998041597e0 * t106 * t100 + 0.234528941479e0 * t106 * t102 - 0.23185843322e-1 * t106 * t104, -params_a_d * t119)
t127 = math.sqrt(0.3e1)
t129 = t46 / t31
t130 = math.sqrt(s0)
t135 = math.sqrt(t129 * t130 / t36 / r0)
t139 = math.exp(-0.98958000000000000000e1 * t127 / t135)
t144 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * ((0.1e1 + params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + 0.5e1 / 0.972e3 * t34 * t40 + t50 * t51 / t36 / t52 / r0 * t62 / 0.576e3 + t91))) * (0.1e1 - t121) + 0.1174e1 * t121) * (0.1e1 - t139))
t146 = lax_cond(t10, t15, -t17)
t147 = lax_cond(t14, t11, t146)
t148 = 0.1e1 + t147
t150 = t148 ** (0.1e1 / 0.3e1)
t152 = lax_cond(t148 <= p_a_zeta_threshold, t23, t150 * t148)
t154 = r1 ** 2
t155 = r1 ** (0.1e1 / 0.3e1)
t156 = t155 ** 2
t158 = 0.1e1 / t156 / t154
t159 = s2 * t158
t162 = s2 ** 2
t163 = t154 ** 2
t169 = t33 * s2 * t158
t172 = math.exp(-0.27e2 / 0.80e2 * t57 * t169)
t188 = (tau1 / t156 / r1 - t159 / 0.8e1) / (t76 + params_a_eta * s2 * t158 / 0.8e1)
t189 = 0.1e1 - t188
t191 = t189 ** 2
t193 = math.exp(-t191 / 0.2e1)
t197 = (0.7e1 / 0.12960e5 * t67 * t169 + t66 * t189 * t193 / 0.100e3) ** 2
t205 = 0.25e1 < t188
t206 = lax_cond(t205, 0.25e1, t188)
t208 = t206 ** 2
t210 = t208 * t206
t212 = t208 ** 2
t221 = lax_cond(t205, t188, 0.25e1)
t225 = math.exp(params_a_c2 / (0.1e1 - t221))
t227 = lax_cond(t188 <= 0.25e1, 0.1e1 - 0.667e0 * t206 - 0.4445555e0 * t208 - 0.663086601049e0 * t210 + 0.1451297044490e1 * t212 - 0.887998041597e0 * t212 * t206 + 0.234528941479e0 * t212 * t208 - 0.23185843322e-1 * t212 * t210, -params_a_d * t225)
t233 = math.sqrt(s2)
t238 = math.sqrt(t129 * t233 / t155 / r1)
t242 = math.exp(-0.98958000000000000000e1 * t127 / t238)
t247 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t152 * t28 * ((0.1e1 + params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + 0.5e1 / 0.972e3 * t34 * t159 + t50 * t162 / t155 / t163 / r1 * t172 / 0.576e3 + t197))) * (0.1e1 - t227) + 0.1174e1 * t227) * (0.1e1 - t242))
res = t144 + t247
