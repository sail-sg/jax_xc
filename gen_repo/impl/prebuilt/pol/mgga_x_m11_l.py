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
t29 = 9 ** (0.1e1 / 0.3e1)
t30 = t29 ** 2
t32 = (0.1e1 / math.pi) ** (0.1e1 / 0.3e1)
t33 = t32 ** 2
t35 = t30 * t33 * p_a_cam_omega
t37 = t2 / t27
t39 = 0.1e1 + t17 <= p_a_zeta_threshold
t41 = 0.1e1 - t17 <= p_a_zeta_threshold
t42 = lax_cond(t41, t15, t17)
t43 = lax_cond(t39, t11, t42)
t44 = 0.1e1 + t43
t46 = t44 ** (0.1e1 / 0.3e1)
t47 = lax_cond(t44 <= p_a_zeta_threshold, t22, t46)
t51 = t35 * t37 / t47 / 0.18e2
t53 = 0.135e1 < t51
t54 = lax_cond(t53, t51, 0.135e1)
t55 = t54 ** 2
t58 = t55 ** 2
t61 = t58 * t55
t64 = t58 ** 2
t76 = t64 ** 2
t80 = lax_cond(t53, 0.135e1, t51)
t81 = math.sqrt(math.pi)
t84 = math.erf(0.1e1 / t80 / 0.2e1)
t86 = t80 ** 2
t89 = math.exp(-0.1e1 / t86 / 0.4e1)
t100 = lax_cond(0.135e1 <= t51, 0.1e1 / t55 / 0.36e2 - 0.1e1 / t58 / 0.960e3 + 0.1e1 / t61 / 0.26880e5 - 0.1e1 / t64 / 0.829440e6 + 0.1e1 / t64 / t55 / 0.28385280e8 - 0.1e1 / t64 / t58 / 0.1073479680e10 + 0.1e1 / t64 / t61 / 0.44590694400e11 - 0.1e1 / t76 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t80 * (t81 * t84 + 0.2e1 * t80 * (t89 - 0.3e1 / 0.2e1 - 0.2e1 * t86 * (t89 - 0.1e1))))
t101 = 6 ** (0.1e1 / 0.3e1)
t102 = math.pi ** 2
t103 = t102 ** (0.1e1 / 0.3e1)
t104 = t103 ** 2
t106 = t101 / t104
t107 = r0 ** 2
t108 = r0 ** (0.1e1 / 0.3e1)
t109 = t108 ** 2
t113 = t106 * s0 / t109 / t107
t118 = 0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91462500000000000000e-2 * t113)
t119 = params_a_a[0]
t120 = params_a_a[1]
t121 = t101 ** 2
t123 = 0.3e1 / 0.10e2 * t121 * t104
t126 = tau0 / t109 / r0
t127 = t123 - t126
t129 = t123 + t126
t130 = 0.1e1 / t129
t132 = params_a_a[2]
t133 = t127 ** 2
t135 = t129 ** 2
t136 = 0.1e1 / t135
t138 = params_a_a[3]
t139 = t133 * t127
t141 = t135 * t129
t142 = 0.1e1 / t141
t144 = params_a_a[4]
t145 = t133 ** 2
t147 = t135 ** 2
t148 = 0.1e1 / t147
t150 = params_a_a[5]
t151 = t145 * t127
t154 = 0.1e1 / t147 / t129
t156 = params_a_a[6]
t157 = t145 * t133
t160 = 0.1e1 / t147 / t135
t162 = params_a_a[7]
t163 = t145 * t139
t166 = 0.1e1 / t147 / t141
t168 = params_a_a[8]
t169 = t145 ** 2
t171 = t147 ** 2
t172 = 0.1e1 / t171
t174 = params_a_a[9]
t175 = t169 * t127
t178 = 0.1e1 / t171 / t129
t180 = params_a_a[10]
t181 = t169 * t133
t184 = 0.1e1 / t171 / t135
t186 = params_a_a[11]
t187 = t169 * t139
t190 = 0.1e1 / t171 / t141
t192 = t119 + t120 * t127 * t130 + t132 * t133 * t136 + t138 * t139 * t142 + t144 * t145 * t148 + t150 * t151 * t154 + t156 * t157 * t160 + t162 * t163 * t166 + t168 * t169 * t172 + t174 * t175 * t178 + t180 * t181 * t184 + t186 * t187 * t190
t195 = math.exp(-0.93189002206715572255e-2 * t113)
t197 = 0.1552e1 - 0.552e0 * t195
t198 = params_a_b[0]
t199 = params_a_b[1]
t202 = params_a_b[2]
t205 = params_a_b[3]
t208 = params_a_b[4]
t211 = params_a_b[5]
t214 = params_a_b[6]
t217 = params_a_b[7]
t220 = params_a_b[8]
t223 = params_a_b[9]
t226 = params_a_b[10]
t229 = params_a_b[11]
t232 = t198 + t199 * t127 * t130 + t202 * t133 * t136 + t205 * t139 * t142 + t208 * t145 * t148 + t211 * t151 * t154 + t214 * t157 * t160 + t217 * t163 * t166 + t220 * t169 * t172 + t223 * t175 * t178 + t226 * t181 * t184 + t229 * t187 * t190
t237 = params_a_c[0]
t238 = params_a_c[1]
t241 = params_a_c[2]
t244 = params_a_c[3]
t247 = params_a_c[4]
t250 = params_a_c[5]
t253 = params_a_c[6]
t256 = params_a_c[7]
t259 = params_a_c[8]
t262 = params_a_c[9]
t265 = params_a_c[10]
t268 = params_a_c[11]
t271 = t237 + t238 * t127 * t130 + t241 * t133 * t136 + t244 * t139 * t142 + t247 * t145 * t148 + t250 * t151 * t154 + t253 * t157 * t160 + t256 * t163 * t166 + t259 * t169 * t172 + t262 * t175 * t178 + t265 * t181 * t184 + t268 * t187 * t190
t273 = params_a_d[0]
t274 = params_a_d[1]
t277 = params_a_d[2]
t280 = params_a_d[3]
t283 = params_a_d[4]
t286 = params_a_d[5]
t289 = params_a_d[6]
t292 = params_a_d[7]
t295 = params_a_d[8]
t298 = params_a_d[9]
t301 = params_a_d[10]
t304 = params_a_d[11]
t307 = t273 + t274 * t127 * t130 + t277 * t133 * t136 + t280 * t139 * t142 + t283 * t145 * t148 + t286 * t151 * t154 + t289 * t157 * t160 + t292 * t163 * t166 + t295 * t169 * t172 + t298 * t175 * t178 + t301 * t181 * t184 + t304 * t187 * t190
t315 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t100 * (t118 * t192 + t197 * t232) + (0.1e1 - t100) * (t118 * t271 + t197 * t307)))
t317 = lax_cond(t10, t15, -t17)
t318 = lax_cond(t14, t11, t317)
t319 = 0.1e1 + t318
t321 = t319 ** (0.1e1 / 0.3e1)
t323 = lax_cond(t319 <= p_a_zeta_threshold, t23, t321 * t319)
t325 = lax_cond(t39, t15, -t17)
t326 = lax_cond(t41, t11, t325)
t327 = 0.1e1 + t326
t329 = t327 ** (0.1e1 / 0.3e1)
t330 = lax_cond(t327 <= p_a_zeta_threshold, t22, t329)
t334 = t35 * t37 / t330 / 0.18e2
t336 = 0.135e1 < t334
t337 = lax_cond(t336, t334, 0.135e1)
t338 = t337 ** 2
t341 = t338 ** 2
t344 = t341 * t338
t347 = t341 ** 2
t359 = t347 ** 2
t363 = lax_cond(t336, 0.135e1, t334)
t366 = math.erf(0.1e1 / t363 / 0.2e1)
t368 = t363 ** 2
t371 = math.exp(-0.1e1 / t368 / 0.4e1)
t382 = lax_cond(0.135e1 <= t334, 0.1e1 / t338 / 0.36e2 - 0.1e1 / t341 / 0.960e3 + 0.1e1 / t344 / 0.26880e5 - 0.1e1 / t347 / 0.829440e6 + 0.1e1 / t347 / t338 / 0.28385280e8 - 0.1e1 / t347 / t341 / 0.1073479680e10 + 0.1e1 / t347 / t344 / 0.44590694400e11 - 0.1e1 / t359 / 0.2021444812800e13, 0.1e1 - 0.8e1 / 0.3e1 * t363 * (t81 * t366 + 0.2e1 * t363 * (t371 - 0.3e1 / 0.2e1 - 0.2e1 * t368 * (t371 - 0.1e1))))
t383 = r1 ** 2
t384 = r1 ** (0.1e1 / 0.3e1)
t385 = t384 ** 2
t389 = t106 * s2 / t385 / t383
t394 = 0.18040e1 - 0.64641600e0 / (0.8040e0 + 0.91462500000000000000e-2 * t389)
t397 = tau1 / t385 / r1
t398 = t123 - t397
t400 = t123 + t397
t401 = 0.1e1 / t400
t403 = t398 ** 2
t405 = t400 ** 2
t406 = 0.1e1 / t405
t408 = t403 * t398
t410 = t405 * t400
t411 = 0.1e1 / t410
t413 = t403 ** 2
t415 = t405 ** 2
t416 = 0.1e1 / t415
t418 = t413 * t398
t421 = 0.1e1 / t415 / t400
t423 = t413 * t403
t426 = 0.1e1 / t415 / t405
t428 = t413 * t408
t431 = 0.1e1 / t415 / t410
t433 = t413 ** 2
t435 = t415 ** 2
t436 = 0.1e1 / t435
t438 = t433 * t398
t441 = 0.1e1 / t435 / t400
t443 = t433 * t403
t446 = 0.1e1 / t435 / t405
t448 = t433 * t408
t451 = 0.1e1 / t435 / t410
t453 = t119 + t120 * t398 * t401 + t132 * t403 * t406 + t138 * t408 * t411 + t144 * t413 * t416 + t150 * t418 * t421 + t156 * t423 * t426 + t162 * t428 * t431 + t168 * t433 * t436 + t174 * t438 * t441 + t180 * t443 * t446 + t186 * t448 * t451
t456 = math.exp(-0.93189002206715572255e-2 * t389)
t458 = 0.1552e1 - 0.552e0 * t456
t481 = t198 + t199 * t398 * t401 + t202 * t403 * t406 + t205 * t408 * t411 + t208 * t413 * t416 + t211 * t418 * t421 + t214 * t423 * t426 + t217 * t428 * t431 + t220 * t433 * t436 + t223 * t438 * t441 + t226 * t443 * t446 + t229 * t448 * t451
t508 = t237 + t238 * t398 * t401 + t241 * t403 * t406 + t244 * t408 * t411 + t247 * t413 * t416 + t250 * t418 * t421 + t253 * t423 * t426 + t256 * t428 * t431 + t259 * t433 * t436 + t262 * t438 * t441 + t265 * t443 * t446 + t268 * t448 * t451
t532 = t273 + t274 * t398 * t401 + t277 * t403 * t406 + t280 * t408 * t411 + t283 * t413 * t416 + t286 * t418 * t421 + t289 * t423 * t426 + t292 * t428 * t431 + t295 * t433 * t436 + t298 * t438 * t441 + t301 * t443 * t446 + t304 * t448 * t451
t540 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t323 * t27 * (t382 * (t394 * t453 + t458 * t481) + (0.1e1 - t382) * (t394 * t508 + t458 * t532)))
res = t315 + t540
