t3 = 3 ** (0.1e1 / 0.3e1)
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = 0.1e1 <= p_a_zeta_threshold
t8 = p_a_zeta_threshold - 0.1e1
t10 = lax_cond(t7, -t8, 0)
t11 = lax_cond(t7, t8, t10)
t12 = 0.1e1 + t11
t13 = t12 <= p_a_zeta_threshold
t14 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t16 = t12 ** (0.1e1 / 0.3e1)
t18 = lax_cond(t13, t14 * p_a_zeta_threshold, t16 * t12)
t19 = r0 ** (0.1e1 / 0.3e1)
t21 = t3 ** 2
t22 = p_a_cam_omega * t21
t23 = math.pi ** 2
t24 = t23 ** (0.1e1 / 0.3e1)
t25 = 0.1e1 / t24
t26 = t22 * t25
t27 = lax_cond(t13, t14, t16)
t28 = 0.1e1 / t27
t29 = 0.1e1 / t19
t30 = t28 * t29
t31 = 6 ** (0.1e1 / 0.3e1)
t32 = t24 ** 2
t33 = 0.1e1 / t32
t34 = t31 * t33
t35 = t34 * s0
t36 = 2 ** (0.1e1 / 0.3e1)
t37 = t36 ** 2
t38 = r0 ** 2
t39 = t19 ** 2
t41 = 0.1e1 / t39 / t38
t42 = t37 * t41
t47 = s0 * t37 * t41
t51 = 0.1e1 / t23
t53 = math.sqrt(s0)
t55 = t38 ** 2
t57 = t53 * s0 / t55
t61 = t31 ** 2
t64 = 0.1e1 / t24 / t23
t66 = s0 ** 2
t71 = t66 * t36 / t19 / t55 / r0
t77 = 0.1e1 / t32 / t23
t84 = t53 * t66 * t37 / t39 / t55 / t38
t88 = t23 ** 2
t89 = 0.1e1 / t88
t91 = t66 * s0
t92 = t55 ** 2
t94 = t91 / t92
t100 = 0.1e1 / t24 / t88
t107 = t53 * t91 * t36 / t19 / t92 / r0
t153 = t66 ** 2
t176 = t35 * t42 * (params_a_a[0] * t31 * t33 * t47 / 0.24e2 + params_a_a[1] * t51 * t57 / 0.24e2 + params_a_a[2] * t61 * t64 * t71 / 0.288e3 + params_a_a[3] * t31 * t77 * t84 / 0.576e3 + params_a_a[4] * t89 * t94 / 0.576e3 + params_a_a[5] * t61 * t100 * t107 / 0.6912e4) / (0.1e1 + params_a_b[0] * t61 * t25 * t53 * t36 / t19 / r0 / 0.12e2 + params_a_b[1] * t31 * t33 * t47 / 0.24e2 + params_a_b[2] * t51 * t57 / 0.24e2 + params_a_b[3] * t61 * t64 * t71 / 0.288e3 + params_a_b[4] * t31 * t77 * t84 / 0.576e3 + params_a_b[5] * t89 * t94 / 0.576e3 + params_a_b[6] * t61 * t100 * t107 / 0.6912e4 + params_a_b[7] * t31 / t32 / t88 * t153 * t37 / t39 / t92 / t38 / 0.13824e5 + params_a_b[8] / t88 / t23 * t53 * t153 / t92 / t55 / 0.13824e5) / 0.24e2
t178 = lax_cond(0.1e-9 < t176, t176, 0.1e-9)
t179 = p_a_cam_omega ** 2
t181 = t27 ** 2
t186 = t179 * t3 * t33 / t181 / t39
t188 = 0.609650e0 + t178 + t186 / 0.3e1
t189 = math.sqrt(t188)
t192 = t26 * t30 / t189
t195 = 0.609650e0 + t178
t207 = 0.1e1 + 0.13006513974354692214e-1 * t35 * t42 / (0.1e1 + t34 * t47 / 0.96e2) + 0.42141105276909202774e1 * t178
t217 = t179 * p_a_cam_omega * t51 / t181 / t27 / r0 / t189 / t188
t221 = t195 ** 2
t228 = t221 * t195
t230 = math.sqrt(t195)
t232 = math.sqrt(math.pi)
t234 = math.sqrt(t178)
t239 = lax_cond(0.0e0 < 0.7572109999e0 + t178, 0.757211e0 + t178, 0.1e-9)
t240 = math.sqrt(t239)
t247 = t179 ** 2
t251 = t181 ** 2
t257 = t188 ** 2
t269 = 0.3e1 * t186
t271 = math.sqrt(0.9e1 * t178 + t269)
t274 = math.sqrt(0.9e1 * t239 + t269)
t282 = t22 * t25 * t28 * t29
t287 = 0.1e1 / (t282 / 0.3e1 + t189)
t289 = math.log((t282 / 0.3e1 + t271 / 0.3e1) * t287)
t295 = math.log((t282 / 0.3e1 + t274 / 0.3e1) * t287)
t302 = lax_cond(r0 / 0.2e1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t3 / t4 * t18 * t19 * (0.757211e0 + 0.47272888888888888889e-1 * (0.1e1 - t192 / 0.3e1) / t195 + 0.26366444444444444444e-1 * t207 * (0.2e1 - t192 + t217 / 0.3e1) / t221 - (0.47459600000000000000e-1 * t207 * t195 + 0.28363733333333333333e-1 * t221 - 0.90865320000000000000e0 * t228 - t230 * t228 * (0.4e1 / 0.5e1 * t232 + 0.12e2 / 0.5e1 * t234 - 0.12e2 / 0.5e1 * t240)) * (0.8e1 - 0.5e1 * t192 + 0.10e2 / 0.3e1 * t217 - t247 * p_a_cam_omega * t3 * t77 / t251 / t27 / t39 / r0 / t189 / t257 / 0.3e1) / t228 / 0.9e1 + 0.2e1 / 0.3e1 * t26 * t30 * (t271 / 0.3e1 - t274 / 0.3e1) + 0.2e1 * t178 * t289 - 0.2e1 * t239 * t295))
res = 0.2e1 * t302
