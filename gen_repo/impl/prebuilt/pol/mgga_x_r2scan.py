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
t30 = 0.20e2 / 0.27e2 + 0.5e1 / 0.3e1 * params_a_eta
t31 = 6 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t33 = math.pi ** 2
t34 = t33 ** (0.1e1 / 0.3e1)
t37 = t32 / t34 / t33
t38 = s0 ** 2
t39 = r0 ** 2
t40 = t39 ** 2
t42 = r0 ** (0.1e1 / 0.3e1)
t46 = params_a_dp2 ** 2
t47 = t46 ** 2
t48 = 0.1e1 / t47
t52 = math.exp(-t37 * t38 / t42 / t40 / r0 * t48 / 0.576e3)
t57 = t34 ** 2
t58 = 0.1e1 / t57
t60 = t42 ** 2
t62 = 0.1e1 / t60 / t39
t70 = params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + (-0.162742215233874e0 * t30 * t52 + 0.10e2 / 0.81e2) * t31 * t58 * s0 * t62 / 0.24e2))
t78 = 0.3e1 / 0.10e2 * t32 * t57
t84 = (tau0 / t60 / r0 - s0 * t62 / 0.8e1) / (t78 + params_a_eta * s0 * t62 / 0.8e1)
t87 = lax_cond(0.0e0 < t84, 0, t84)
t92 = math.exp(-params_a_c1 * t87 / (0.1e1 - t87))
t94 = 0.25e1 < t84
t95 = lax_cond(t94, 0.25e1, t84)
t97 = t95 ** 2
t99 = t97 * t95
t101 = t97 ** 2
t110 = lax_cond(t94, t84, 0.25e1)
t114 = math.exp(params_a_c2 / (0.1e1 - t110))
t116 = lax_cond(t84 <= 0.25e1, 0.1e1 - 0.667e0 * t95 - 0.4445555e0 * t97 - 0.663086601049e0 * t99 + 0.1451297044490e1 * t101 - 0.887998041597e0 * t101 * t95 + 0.234528941479e0 * t101 * t97 - 0.23185843322e-1 * t101 * t99, -params_a_d * t114)
t117 = lax_cond(t84 <= 0.0e0, t92, t116)
t122 = math.sqrt(0.3e1)
t124 = t32 / t34
t125 = math.sqrt(s0)
t130 = math.sqrt(t124 * t125 / t42 / r0)
t134 = math.exp(-0.98958000000000000000e1 * t122 / t130)
t139 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t28 * (0.1e1 + t70 + t117 * (0.174e0 - t70)) * (0.1e1 - t134))
t141 = lax_cond(t10, t15, -t17)
t142 = lax_cond(t14, t11, t141)
t143 = 0.1e1 + t142
t145 = t143 ** (0.1e1 / 0.3e1)
t147 = lax_cond(t143 <= p_a_zeta_threshold, t23, t145 * t143)
t149 = s2 ** 2
t150 = r1 ** 2
t151 = t150 ** 2
t153 = r1 ** (0.1e1 / 0.3e1)
t160 = math.exp(-t37 * t149 / t153 / t151 / r1 * t48 / 0.576e3)
t166 = t153 ** 2
t168 = 0.1e1 / t166 / t150
t176 = params_a_k1 * (0.1e1 - params_a_k1 / (params_a_k1 + (-0.162742215233874e0 * t30 * t160 + 0.10e2 / 0.81e2) * t31 * t58 * s2 * t168 / 0.24e2))
t188 = (tau1 / t166 / r1 - s2 * t168 / 0.8e1) / (t78 + params_a_eta * s2 * t168 / 0.8e1)
t191 = lax_cond(0.0e0 < t188, 0, t188)
t196 = math.exp(-params_a_c1 * t191 / (0.1e1 - t191))
t198 = 0.25e1 < t188
t199 = lax_cond(t198, 0.25e1, t188)
t201 = t199 ** 2
t203 = t201 * t199
t205 = t201 ** 2
t214 = lax_cond(t198, t188, 0.25e1)
t218 = math.exp(params_a_c2 / (0.1e1 - t214))
t220 = lax_cond(t188 <= 0.25e1, 0.1e1 - 0.667e0 * t199 - 0.4445555e0 * t201 - 0.663086601049e0 * t203 + 0.1451297044490e1 * t205 - 0.887998041597e0 * t205 * t199 + 0.234528941479e0 * t205 * t201 - 0.23185843322e-1 * t205 * t203, -params_a_d * t218)
t221 = lax_cond(t188 <= 0.0e0, t196, t220)
t226 = math.sqrt(s2)
t231 = math.sqrt(t124 * t226 / t153 / r1)
t235 = math.exp(-0.98958000000000000000e1 * t122 / t231)
t240 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t147 * t28 * (0.1e1 + t176 + t221 * (0.174e0 - t176)) * (0.1e1 - t235))
res = t139 + t240
