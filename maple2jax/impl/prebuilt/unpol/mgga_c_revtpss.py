t2 = lax_cond(0 < 0, 0, 0)
t4 = params_a_C0_c[0]
t9 = 0.1e1 <= p_a_zeta_threshold
t10 = p_a_zeta_threshold - 0.1e1
t12 = lax_cond(t9, -t10, 0)
t13 = lax_cond(t9, t10, t12)
t14 = t13 ** 2
t16 = 2 ** (0.1e1 / 0.3e1)
t17 = t16 ** 2
t18 = s0 * t17
t19 = r0 ** 2
t20 = r0 ** (0.1e1 / 0.3e1)
t21 = t20 ** 2
t23 = 0.1e1 / t21 / t19
t24 = 0.1e1 + t13
t25 = t24 / 0.2e1
t26 = t25 ** (0.1e1 / 0.3e1)
t27 = t26 ** 2
t31 = 0.1e1 - t13
t32 = t31 / 0.2e1
t33 = t32 ** (0.1e1 / 0.3e1)
t34 = t33 ** 2
t41 = 3 ** (0.1e1 / 0.3e1)
t42 = math.pi ** 2
t43 = t42 ** (0.1e1 / 0.3e1)
t44 = t43 ** 2
t47 = t24 ** (0.1e1 / 0.3e1)
t48 = t47 * t24
t50 = t31 ** (0.1e1 / 0.3e1)
t51 = t50 * t31
t58 = (0.1e1 + (0.1e1 - t14) * (t18 * t23 * t27 * t25 + t18 * t23 * t34 * t32 - s0 * t23) * t41 / t44 * (0.1e1 / t48 + 0.1e1 / t51) / 0.24e2) ** 2
t59 = t58 ** 2
t62 = lax_cond(-t2 <= -0.999999999999e0, t4 + params_a_C0_c[1] + params_a_C0_c[2] + params_a_C0_c[3], t4 / t59)
t68 = s0 / r0 / tau0 / 0.8e1
t70 = lax_cond(0.1e1 < t68, 1, t68)
t71 = t70 ** 2
t75 = jnp.logical_or(r0 / 0.2e1 <= p_a_dens_threshold, t9)
t77 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t78 = t41 * t77
t79 = 4 ** (0.1e1 / 0.3e1)
t80 = t79 ** 2
t81 = 0.1e1 / t20
t83 = t78 * t80 * t81
t86 = math.sqrt(t83)
t89 = t83 ** 0.15e1
t91 = t41 ** 2
t92 = t77 ** 2
t93 = t91 * t92
t94 = 0.1e1 / t21
t96 = t93 * t79 * t94
t102 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t86 + 0.89690000000000000000e0 * t83 + 0.20477500000000000000e0 * t89 + 0.12323500000000000000e0 * t96))
t103 = (0.1e1 + 0.53425000000000000000e-1 * t83) * t102
t105 = t14 ** 2
t106 = t24 <= p_a_zeta_threshold
t107 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t108 = t107 * p_a_zeta_threshold
t109 = lax_cond(t106, t108, t48)
t110 = t31 <= p_a_zeta_threshold
t111 = lax_cond(t110, t108, t51)
t112 = t109 + t111 - 0.2e1
t116 = 0.1e1 / (0.2e1 * t16 - 0.2e1)
t127 = math.log(0.1e1 + 0.32163958997385070134e2 / (0.70594500000000000000e1 * t86 + 0.15494250000000000000e1 * t83 + 0.42077500000000000000e0 * t89 + 0.15629250000000000000e0 * t96))
t130 = 0.621814e-1 * t103
t141 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t86 + 0.90577500000000000000e0 * t83 + 0.11003250000000000000e0 * t89 + 0.12417750000000000000e0 * t96))
t142 = (0.1e1 + 0.27812500000000000000e-1 * t83) * t141
t146 = t105 * t112 * t116 * (-0.3109070e-1 * (0.1e1 + 0.51370000000000000000e-1 * t83) * t127 + t130 - 0.19751673498613801407e-1 * t142)
t149 = t112 * t116 * t142
t151 = math.log(0.2e1)
t152 = 0.1e1 - t151
t154 = t152 / t42
t155 = t107 ** 2
t156 = t47 ** 2
t157 = lax_cond(t106, t155, t156)
t158 = t50 ** 2
t159 = lax_cond(t110, t155, t158)
t161 = t157 / 0.2e1 + t159 / 0.2e1
t162 = t161 ** 2
t163 = t162 * t161
t169 = (0.1e1 + 0.25000000000000000000e-1 * t83) / (0.1e1 + 0.44450000000000000000e-1 * t83)
t172 = s0 / t20 / t19
t173 = t172 * t16
t177 = 0.1e1 / t77 * t79
t181 = 0.1e1 / t152
t182 = t169 * t181
t183 = 0.19751673498613801407e-1 * t149
t189 = math.exp(-(-t130 + t146 + t183) * t181 * t42 / t163)
t192 = t42 / (t189 - 0.1e1)
t193 = s0 ** 2
t196 = t19 ** 2
t198 = 0.1e1 / t21 / t196
t199 = t198 * t17
t200 = t162 ** 2
t203 = 0.1e1 / t92
t205 = t41 * t203 * t80
t209 = t173 / t162 * t91 * t177 / 0.96e2 + 0.21720231316129303386e-4 * t182 * t192 * t193 * t199 / t200 * t205
t211 = t181 * t42
t221 = math.log(0.1e1 + 0.66724550603149220e-1 * t169 * t209 * t211 / (0.1e1 + 0.66724550603149220e-1 * t182 * t192 * t209))
t223 = t154 * t163 * t221
t225 = -0.31090700000000000000e-1 * t103 + t146 / 0.2e1 + 0.98758367493069007035e-2 * t149 + t223 / 0.2e1
t226 = -t130 + t146 + t183 + t223
t227 = t78 * t80
t228 = t81 * t16
t230 = (0.1e1 / t24) ** (0.1e1 / 0.3e1)
t232 = t227 * t228 * t230
t235 = math.sqrt(t232)
t238 = t232 ** 0.15e1
t240 = t93 * t79
t241 = t94 * t17
t242 = t230 ** 2
t244 = t240 * t241 * t242
t250 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t235 + 0.89690000000000000000e0 * t232 + 0.20477500000000000000e0 * t238 + 0.12323500000000000000e0 * t244))
t252 = 0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t232) * t250
t253 = 0.2e1 <= p_a_zeta_threshold
t255 = lax_cond(t253, t108, 0.2e1 * t16)
t256 = 0.0e0 <= p_a_zeta_threshold
t257 = lax_cond(t256, t108, 0)
t259 = (t255 + t257 - 0.2e1) * t116
t270 = math.log(0.1e1 + 0.32163958997385070134e2 / (0.70594500000000000000e1 * t235 + 0.15494250000000000000e1 * t232 + 0.42077500000000000000e0 * t238 + 0.15629250000000000000e0 * t244))
t283 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t235 + 0.90577500000000000000e0 * t232 + 0.11003250000000000000e0 * t238 + 0.12417750000000000000e0 * t244))
t284 = (0.1e1 + 0.27812500000000000000e-1 * t232) * t283
t287 = t259 * (-0.3109070e-1 * (0.1e1 + 0.51370000000000000000e-1 * t232) * t270 + t252 - 0.19751673498613801407e-1 * t284)
t289 = 0.19751673498613801407e-1 * t259 * t284
t290 = lax_cond(t253, t155, t17)
t291 = lax_cond(t256, t155, 0)
t293 = t290 / 0.2e1 + t291 / 0.2e1
t294 = t293 ** 2
t295 = t294 * t293
t301 = (0.1e1 + 0.25000000000000000000e-1 * t232) / (0.1e1 + 0.44450000000000000000e-1 * t232)
t304 = t172 / t294 * t91
t310 = t301 * t181
t314 = t42 / t295
t316 = math.exp(-(-t252 + t287 + t289) * t181 * t314)
t319 = t42 / (t316 - 0.1e1)
t322 = t294 ** 2
t325 = t198 / t322 * t41
t326 = t203 * t80
t333 = t304 * t177 * t17 / t230 / 0.96e2 + 0.43440462632258606772e-4 * t310 * t319 * t193 * t325 * t326 * t16 / t242
t344 = math.log(0.1e1 + 0.66724550603149220e-1 * t301 * t333 * t211 / (0.1e1 + 0.66724550603149220e-1 * t310 * t319 * t333))
t347 = t154 * t295 * t344 - t252 + t287 + t289
t349 = lax_cond(t226 < t347, t347, t226)
t352 = lax_cond(t75, t225, t349 * t24 / 0.2e1)
t354 = (0.1e1 / t31) ** (0.1e1 / 0.3e1)
t356 = t227 * t228 * t354
t359 = math.sqrt(t356)
t362 = t356 ** 0.15e1
t364 = t354 ** 2
t366 = t240 * t241 * t364
t372 = math.log(0.1e1 + 0.16081979498692535067e2 / (0.37978500000000000000e1 * t359 + 0.89690000000000000000e0 * t356 + 0.20477500000000000000e0 * t362 + 0.12323500000000000000e0 * t366))
t374 = 0.621814e-1 * (0.1e1 + 0.53425000000000000000e-1 * t356) * t372
t385 = math.log(0.1e1 + 0.32163958997385070134e2 / (0.70594500000000000000e1 * t359 + 0.15494250000000000000e1 * t356 + 0.42077500000000000000e0 * t362 + 0.15629250000000000000e0 * t366))
t398 = math.log(0.1e1 + 0.29608749977793437516e2 / (0.51785000000000000000e1 * t359 + 0.90577500000000000000e0 * t356 + 0.11003250000000000000e0 * t362 + 0.12417750000000000000e0 * t366))
t399 = (0.1e1 + 0.27812500000000000000e-1 * t356) * t398
t402 = t259 * (-0.3109070e-1 * (0.1e1 + 0.51370000000000000000e-1 * t356) * t385 + t374 - 0.19751673498613801407e-1 * t399)
t404 = 0.19751673498613801407e-1 * t259 * t399
t410 = (0.1e1 + 0.25000000000000000000e-1 * t356) / (0.1e1 + 0.44450000000000000000e-1 * t356)
t416 = t410 * t181
t420 = math.exp(-(-t374 + t402 + t404) * t181 * t314)
t423 = t42 / (t420 - 0.1e1)
t432 = t304 * t177 * t17 / t354 / 0.96e2 + 0.43440462632258606772e-4 * t416 * t423 * t193 * t325 * t326 * t16 / t364
t443 = math.log(0.1e1 + 0.66724550603149220e-1 * t410 * t432 * t211 / (0.1e1 + 0.66724550603149220e-1 * t416 * t423 * t432))
t446 = t154 * t295 * t443 - t374 + t402 + t404
t448 = lax_cond(t226 < t446, t446, t226)
t451 = lax_cond(t75, t225, t448 * t31 / 0.2e1)
t456 = lax_cond(t9, t108, 1)
t461 = 0.19751673498613801407e-1 * (0.2e1 * t456 - 0.2e1) * t116 * t142
t462 = lax_cond(t9, t155, 1)
t463 = t462 ** 2
t464 = t463 * t462
t475 = math.exp(-(-t130 + t461) * t181 * t42 / t464)
t478 = t42 / (t475 - 0.1e1)
t481 = t463 ** 2
t487 = t173 / t463 * t91 * t177 / 0.96e2 + 0.21720231316129303386e-4 * t182 * t478 * t193 * t199 / t481 * t205
t498 = math.log(0.1e1 + 0.66724550603149220e-1 * t169 * t487 * t211 / (0.1e1 + 0.66724550603149220e-1 * t182 * t478 * t487))
t503 = -(0.1e1 + t62) * t71 * (t352 + t451) + (t62 * t71 + 0.1e1) * (t154 * t464 * t498 - t130 + t461)
res = t503 * (params_a_d * t503 * t71 * t70 + 0.1e1)
