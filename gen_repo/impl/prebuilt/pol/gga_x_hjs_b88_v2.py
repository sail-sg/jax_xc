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
t29 = t2 ** 2
t30 = p_a_cam_omega * t29
t31 = math.pi ** 2
t32 = t31 ** (0.1e1 / 0.3e1)
t33 = 0.1e1 / t32
t34 = t30 * t33
t36 = 0.1e1 + t17 <= p_a_zeta_threshold
t38 = 0.1e1 - t17 <= p_a_zeta_threshold
t39 = lax_cond(t38, t15, t17)
t40 = lax_cond(t36, t11, t39)
t41 = 0.1e1 + t40
t43 = t41 ** (0.1e1 / 0.3e1)
t44 = lax_cond(t41 <= p_a_zeta_threshold, t22, t43)
t45 = 0.1e1 / t44
t46 = 0.1e1 / t27
t47 = t45 * t46
t48 = 6 ** (0.1e1 / 0.3e1)
t49 = t48 ** 2
t50 = t49 * t33
t51 = math.sqrt(s0)
t52 = r0 ** (0.1e1 / 0.3e1)
t58 = math.exp(-t50 * t51 / t52 / r0 / 0.12e2)
t59 = math.exp(20)
t61 = 0.1e1 / (t59 - 0.1e1)
t64 = 0.1e1 / (0.1e1 + t61)
t66 = math.log((t58 + t61) * t64)
t67 = t66 ** 2
t68 = params_a_a[0]
t70 = params_a_a[1]
t71 = t67 * t66
t73 = params_a_a[2]
t74 = t67 ** 2
t76 = params_a_a[3]
t77 = t74 * t66
t79 = params_a_a[4]
t80 = t74 * t67
t82 = params_a_a[5]
t83 = t74 * t71
t87 = params_a_b[0]
t89 = params_a_b[1]
t91 = params_a_b[2]
t93 = params_a_b[3]
t95 = params_a_b[4]
t97 = params_a_b[5]
t99 = params_a_b[6]
t101 = params_a_b[7]
t102 = t74 ** 2
t104 = params_a_b[8]
t109 = t67 * (t68 * t67 - t70 * t71 + t73 * t74 - t76 * t77 + t79 * t80 - t82 * t83) / (-t104 * t102 * t66 + t101 * t102 - t87 * t66 + t89 * t67 - t91 * t71 + t93 * t74 - t95 * t77 + t97 * t80 - t99 * t83 + 0.1e1)
t111 = lax_cond(0.1e-9 < t109, t109, 0.1e-9)
t112 = p_a_cam_omega ** 2
t113 = t112 * t2
t114 = t32 ** 2
t115 = 0.1e1 / t114
t116 = t44 ** 2
t119 = t27 ** 2
t120 = 0.1e1 / t119
t122 = t113 * t115 / t116 * t120
t124 = 0.609650e0 + t111 + t122 / 0.3e1
t125 = math.sqrt(t124)
t128 = t34 * t47 / t125
t131 = 0.609650e0 + t111
t141 = 0.1e1 + 0.31215633538451261314e0 * t67 / (0.1e1 + t67 / 0.4e1) + 0.42141105276909202774e1 * t111
t144 = t112 * p_a_cam_omega / t31
t151 = t144 / t116 / t44 * t7 / t125 / t124
t155 = t131 ** 2
t162 = t155 * t131
t164 = math.sqrt(t131)
t166 = math.sqrt(math.pi)
t167 = 0.4e1 / 0.5e1 * t166
t168 = math.sqrt(t111)
t173 = lax_cond(0.0e0 < 0.7572109999e0 + t111, 0.757211e0 + t111, 0.1e-9)
t174 = math.sqrt(t173)
t181 = t112 ** 2
t186 = t181 * p_a_cam_omega * t2 / t114 / t31
t187 = t116 ** 2
t191 = 0.1e1 / t119 / t6
t193 = t124 ** 2
t205 = 0.3e1 * t122
t207 = math.sqrt(0.9e1 * t111 + t205)
t210 = math.sqrt(0.9e1 * t173 + t205)
t218 = t30 * t33 * t45 * t46
t223 = 0.1e1 / (t218 / 0.3e1 + t125)
t225 = math.log((t218 / 0.3e1 + t207 / 0.3e1) * t223)
t231 = math.log((t218 / 0.3e1 + t210 / 0.3e1) * t223)
t238 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (0.757211e0 + 0.47272888888888888889e-1 * (0.1e1 - t128 / 0.3e1) / t131 + 0.26366444444444444444e-1 * t141 * (0.2e1 - t128 + t151 / 0.3e1) / t155 - (0.47459600000000000000e-1 * t141 * t131 + 0.28363733333333333333e-1 * t155 - 0.90865320000000000000e0 * t162 - t164 * t162 * (t167 + 0.12e2 / 0.5e1 * t168 - 0.12e2 / 0.5e1 * t174)) * (0.8e1 - 0.5e1 * t128 + 0.10e2 / 0.3e1 * t151 - t186 / t187 / t44 * t191 / t125 / t193 / 0.3e1) / t162 / 0.9e1 + 0.2e1 / 0.3e1 * t34 * t47 * (t207 / 0.3e1 - t210 / 0.3e1) + 0.2e1 * t111 * t225 - 0.2e1 * t173 * t231))
t240 = lax_cond(t10, t15, -t17)
t241 = lax_cond(t14, t11, t240)
t242 = 0.1e1 + t241
t244 = t242 ** (0.1e1 / 0.3e1)
t246 = lax_cond(t242 <= p_a_zeta_threshold, t23, t244 * t242)
t248 = lax_cond(t36, t15, -t17)
t249 = lax_cond(t38, t11, t248)
t250 = 0.1e1 + t249
t252 = t250 ** (0.1e1 / 0.3e1)
t253 = lax_cond(t250 <= p_a_zeta_threshold, t22, t252)
t254 = 0.1e1 / t253
t255 = t254 * t46
t256 = math.sqrt(s2)
t257 = r1 ** (0.1e1 / 0.3e1)
t263 = math.exp(-t50 * t256 / t257 / r1 / 0.12e2)
t266 = math.log((t263 + t61) * t64)
t267 = t266 ** 2
t269 = t267 * t266
t271 = t267 ** 2
t273 = t271 * t266
t275 = t271 * t267
t277 = t271 * t269
t288 = t271 ** 2
t294 = t267 * (t68 * t267 - t70 * t269 + t73 * t271 - t76 * t273 + t79 * t275 - t82 * t277) / (-t104 * t288 * t266 + t101 * t288 - t87 * t266 + t89 * t267 - t91 * t269 + t93 * t271 - t95 * t273 + t97 * t275 - t99 * t277 + 0.1e1)
t296 = lax_cond(0.1e-9 < t294, t294, 0.1e-9)
t297 = t253 ** 2
t301 = t113 * t115 / t297 * t120
t303 = 0.609650e0 + t296 + t301 / 0.3e1
t304 = math.sqrt(t303)
t307 = t34 * t255 / t304
t310 = 0.609650e0 + t296
t320 = 0.1e1 + 0.31215633538451261314e0 * t267 / (0.1e1 + t267 / 0.4e1) + 0.42141105276909202774e1 * t296
t327 = t144 / t297 / t253 * t7 / t304 / t303
t331 = t310 ** 2
t338 = t331 * t310
t340 = math.sqrt(t310)
t342 = math.sqrt(t296)
t347 = lax_cond(0.0e0 < 0.7572109999e0 + t296, 0.757211e0 + t296, 0.1e-9)
t348 = math.sqrt(t347)
t355 = t297 ** 2
t359 = t303 ** 2
t371 = 0.3e1 * t301
t373 = math.sqrt(0.9e1 * t296 + t371)
t376 = math.sqrt(0.9e1 * t347 + t371)
t384 = t30 * t33 * t254 * t46
t389 = 0.1e1 / (t384 / 0.3e1 + t304)
t391 = math.log((t384 / 0.3e1 + t373 / 0.3e1) * t389)
t397 = math.log((t384 / 0.3e1 + t376 / 0.3e1) * t389)
t404 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t246 * t27 * (0.757211e0 + 0.47272888888888888889e-1 * (0.1e1 - t307 / 0.3e1) / t310 + 0.26366444444444444444e-1 * t320 * (0.2e1 - t307 + t327 / 0.3e1) / t331 - (0.47459600000000000000e-1 * t320 * t310 + 0.28363733333333333333e-1 * t331 - 0.90865320000000000000e0 * t338 - t340 * t338 * (t167 + 0.12e2 / 0.5e1 * t342 - 0.12e2 / 0.5e1 * t348)) * (0.8e1 - 0.5e1 * t307 + 0.10e2 / 0.3e1 * t327 - t186 / t355 / t253 * t191 / t304 / t359 / 0.3e1) / t338 / 0.9e1 + 0.2e1 / 0.3e1 * t34 * t255 * (t373 / 0.3e1 - t376 / 0.3e1) + 0.2e1 * t296 * t391 - 0.2e1 * t347 * t397))
res = t238 + t404
