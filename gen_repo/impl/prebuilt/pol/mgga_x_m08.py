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
t29 = 6 ** (0.1e1 / 0.3e1)
t30 = math.pi ** 2
t31 = t30 ** (0.1e1 / 0.3e1)
t32 = t31 ** 2
t34 = t29 / t32
t35 = r0 ** 2
t36 = r0 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t41 = t34 * s0 / t37 / t35
t47 = params_a_a[0]
t48 = params_a_a[1]
t49 = t29 ** 2
t51 = 0.3e1 / 0.10e2 * t49 * t32
t54 = tau0 / t37 / r0
t55 = t51 - t54
t57 = t51 + t54
t58 = 0.1e1 / t57
t60 = params_a_a[2]
t61 = t55 ** 2
t63 = t57 ** 2
t64 = 0.1e1 / t63
t66 = params_a_a[3]
t67 = t61 * t55
t69 = t63 * t57
t70 = 0.1e1 / t69
t72 = params_a_a[4]
t73 = t61 ** 2
t75 = t63 ** 2
t76 = 0.1e1 / t75
t78 = params_a_a[5]
t79 = t73 * t55
t82 = 0.1e1 / t75 / t57
t84 = params_a_a[6]
t85 = t73 * t61
t88 = 0.1e1 / t75 / t63
t90 = params_a_a[7]
t91 = t73 * t67
t94 = 0.1e1 / t75 / t69
t96 = params_a_a[8]
t97 = t73 ** 2
t99 = t75 ** 2
t100 = 0.1e1 / t99
t102 = params_a_a[9]
t103 = t97 * t55
t106 = 0.1e1 / t99 / t57
t108 = params_a_a[10]
t109 = t97 * t61
t112 = 0.1e1 / t99 / t63
t114 = params_a_a[11]
t115 = t97 * t67
t118 = 0.1e1 / t99 / t69
t120 = t47 + t48 * t55 * t58 + t60 * t61 * t64 + t66 * t67 * t70 + t72 * t73 * t76 + t78 * t79 * t82 + t84 * t85 * t88 + t90 * t91 * t94 + t96 * t97 * t100 + t102 * t103 * t106 + t108 * t109 * t112 + t114 * t115 * t118
t123 = math.exp(-0.93189002206715572255e-2 * t41)
t126 = params_a_b[0]
t127 = params_a_b[1]
t130 = params_a_b[2]
t133 = params_a_b[3]
t136 = params_a_b[4]
t139 = params_a_b[5]
t142 = params_a_b[6]
t145 = params_a_b[7]
t148 = params_a_b[8]
t151 = params_a_b[9]
t154 = params_a_b[10]
t157 = params_a_b[11]
t160 = t126 + t127 * t55 * t58 + t130 * t61 * t64 + t133 * t67 * t70 + t136 * t73 * t76 + t139 * t79 * t82 + t142 * t85 * t88 + t145 * t91 * t94 + t148 * t97 * t100 + t151 * t103 * t106 + t154 * t109 * t112 + t157 * t115 * t118
t166 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91462500000000000000e-2 * t41)) * t120 + (0.1552e1 - 0.552e0 * t123) * t160))
t168 = lax_cond(t10, t15, -t17)
t169 = lax_cond(t14, t11, t168)
t170 = 0.1e1 + t169
t172 = t170 ** (0.1e1 / 0.3e1)
t174 = lax_cond(t170 <= p_a_zeta_threshold, t23, t172 * t170)
t176 = r1 ** 2
t177 = r1 ** (0.1e1 / 0.3e1)
t178 = t177 ** 2
t182 = t34 * s2 / t178 / t176
t190 = tau1 / t178 / r1
t191 = t51 - t190
t193 = t51 + t190
t194 = 0.1e1 / t193
t196 = t191 ** 2
t198 = t193 ** 2
t199 = 0.1e1 / t198
t201 = t196 * t191
t203 = t198 * t193
t204 = 0.1e1 / t203
t206 = t196 ** 2
t208 = t198 ** 2
t209 = 0.1e1 / t208
t211 = t206 * t191
t214 = 0.1e1 / t208 / t193
t216 = t206 * t196
t219 = 0.1e1 / t208 / t198
t221 = t206 * t201
t224 = 0.1e1 / t208 / t203
t226 = t206 ** 2
t228 = t208 ** 2
t229 = 0.1e1 / t228
t231 = t226 * t191
t234 = 0.1e1 / t228 / t193
t236 = t226 * t196
t239 = 0.1e1 / t228 / t198
t241 = t226 * t201
t244 = 0.1e1 / t228 / t203
t246 = t47 + t48 * t191 * t194 + t60 * t196 * t199 + t66 * t201 * t204 + t72 * t206 * t209 + t78 * t211 * t214 + t84 * t216 * t219 + t90 * t221 * t224 + t96 * t226 * t229 + t102 * t231 * t234 + t108 * t236 * t239 + t114 * t241 * t244
t249 = math.exp(-0.93189002206715572255e-2 * t182)
t274 = t126 + t127 * t191 * t194 + t130 * t196 * t199 + t133 * t201 * t204 + t136 * t206 * t209 + t139 * t211 * t214 + t142 * t216 * t219 + t145 * t221 * t224 + t148 * t226 * t229 + t151 * t231 * t234 + t154 * t236 * t239 + t157 * t241 * t244
t280 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t174 * t27 * ((0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91462500000000000000e-2 * t182)) * t246 + (0.1552e1 - 0.552e0 * t249) * t274))
res = t166 + t280
