t1 = 3 ** (0.1e1 / 0.3e1)
t2 = 0.1e1 / math.pi
t3 = t2 ** (0.1e1 / 0.3e1)
t4 = t1 * t3
t5 = 4 ** (0.1e1 / 0.3e1)
t6 = t5 ** 2
t7 = r0 + r1
t8 = t7 ** (0.1e1 / 0.3e1)
t9 = 0.1e1 / t8
t11 = t4 * t6 * t9
t14 = math.sqrt(t11)
t17 = t11 ** 0.15e1
t19 = t1 ** 2
t20 = t3 ** 2
t21 = t19 * t20
t22 = t8 ** 2
t23 = 0.1e1 / t22
t25 = t21 * t5 * t23
t31 = math.log(0.1e1 + 0.16081824322151104822e2 / (0.37978500000000000000e1 * t14 + 0.89690000000000000000e0 * t11 + 0.20477500000000000000e0 * t17 + 0.12323500000000000000e0 * t25))
t33 = 0.62182e-1 * (0.1e1 + 0.53425000000000000000e-1 * t11) * t31
t35 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t36 = t35 * p_a_zeta_threshold
t37 = lax_cond(0.1e1 <= p_a_zeta_threshold, t36, 1)
t40 = 2 ** (0.1e1 / 0.3e1)
t43 = 0.1e1 / (0.2e1 * t40 - 0.2e1)
t44 = (0.2e1 * t37 - 0.2e1) * t43
t55 = math.log(0.1e1 + 0.29608574643216675549e2 / (0.51785000000000000000e1 * t14 + 0.90577500000000000000e0 * t11 + 0.11003250000000000000e0 * t17 + 0.12417750000000000000e0 * t25))
t56 = (0.1e1 + 0.27812500000000000000e-1 * t11) * t55
t59 = -t33 + 0.19751789702565206229e-1 * t44 * t56
t61 = s0 + 0.2e1 * s1 + s2
t63 = t7 ** 2
t65 = 0.1e1 / t8 / t63
t67 = 0.1e1 / t3
t68 = t19 * t67
t71 = lax_cond(0.0e0 < t59, t59, -t59)
t78 = math.log(0.1e1 + t40 * t61 * t65 * t68 * t5 / t71 / 0.96e2)
t84 = t1 / t20
t85 = t84 * t6
t87 = 0.1e1 / t8 / t7
t88 = 0.1e1 / t7
t89 = t2 * t88
t95 = 0.1e1 + t14 * (0.107924e1 + 0.39640000000000000000e-1 * t14 + 0.12382500000000000000e-1 * t11) / 0.2e1
t96 = t95 ** 2
t102 = t1 * t3 * t2
t107 = t19 * t20 * t2
t109 = 0.1e1 / t22 / t7
t113 = math.pi ** 2
t114 = 0.1e1 / t113
t115 = 0.1e1 / t63
t116 = t114 * t115
t119 = t1 * t3 * t114
t123 = -0.18780000000000000000e-1 * t89 + 0.13173750000000000000e-2 * t102 * t6 * t87 - 0.23775000000000000000e-3 * t107 * t5 * t109 + 0.63900000000000000000e-4 * t116 - 0.54014062500000000000e-6 * t119 * t6 * t65
t125 = 0.36798313500000000000e-2 * t89 / t96 - t59 * t123
t127 = 4 ** (0.1e1 / 0.6e1)
t129 = t14 * t11
t131 = 0.1e1 / t95
t135 = t59 ** 2
t138 = 0.1e1 / (0.19711288999999999999e-2 * t84 * t127 * t22 * t129 * t131 - 0.2e1 * t135)
t141 = t85 * t87 * t125 * t138 * t61
t144 = math.sqrt(0.4e1)
t149 = t6 * t22
t156 = (0.61912500000000000000e-2 * t59 * t144 * t129 * t131 - 0.79593333333333333331e-1 * t84 * t149 * t123) * t138 * t61 * t115
t159 = t61 ** 2
t160 = t63 ** 2
t163 = t125 * t138 * t159 / t160
t167 = (t59 / (0.1e1 + 0.66725e-1 * t78) + 0.99491666666666666664e-2 * t141) / (0.1e1 + t156 / 0.8e1 - t163 / 0.64e2)
t169 = (r0 - r1) * t88
t170 = 0.1e1 + t169
t171 = t170 <= p_a_zeta_threshold
t172 = t170 ** (0.1e1 / 0.3e1)
t174 = lax_cond(t171, t36, t172 * t170)
t175 = 0.1e1 - t169
t176 = t175 <= p_a_zeta_threshold
t177 = t175 ** (0.1e1 / 0.3e1)
t179 = lax_cond(t176, t36, t177 * t175)
t184 = lax_cond(0.2e1 <= p_a_zeta_threshold, t36, 0.2e1 * t40)
t186 = lax_cond(0.0e0 <= p_a_zeta_threshold, t36, 0)
t188 = (t184 + t186 - 0.2e1) * t43
t199 = math.log(0.1e1 + 0.32164683177870697974e2 / (0.70594500000000000000e1 * t14 + 0.15494250000000000000e1 * t11 + 0.42077500000000000000e0 * t17 + 0.15629250000000000000e0 * t25))
t207 = -t33 + t188 * (-0.31090e-1 * (0.1e1 + 0.51370000000000000000e-1 * t11) * t199 + t33 - 0.19751789702565206229e-1 * t56) + 0.19751789702565206229e-1 * t188 * t56
t210 = t67 * t5
t213 = lax_cond(0.0e0 < t207, t207, -t207)
t219 = math.log(0.1e1 + t61 * t65 * t19 * t210 / t213 / 0.96e2)
t236 = lax_cond(t171, p_a_zeta_threshold, t170)
t238 = t4 * t6
t239 = t9 * t40
t240 = 0.1e1 / t170
t241 = t240 ** (0.1e1 / 0.3e1)
t243 = t238 * t239 * t241
t246 = math.sqrt(t243)
t249 = t243 ** 0.15e1
t251 = t21 * t5
t252 = t40 ** 2
t253 = t23 * t252
t254 = t241 ** 2
t256 = t251 * t253 * t254
t262 = math.log(0.1e1 + 0.16081824322151104822e2 / (0.37978500000000000000e1 * t246 + 0.89690000000000000000e0 * t243 + 0.20477500000000000000e0 * t249 + 0.12323500000000000000e0 * t256))
t264 = 0.62182e-1 * (0.1e1 + 0.53425000000000000000e-1 * t243) * t262
t275 = math.log(0.1e1 + 0.29608574643216675549e2 / (0.51785000000000000000e1 * t246 + 0.90577500000000000000e0 * t243 + 0.11003250000000000000e0 * t249 + 0.12417750000000000000e0 * t256))
t276 = (0.1e1 + 0.27812500000000000000e-1 * t243) * t275
t279 = -t264 + 0.19751789702565206229e-1 * t44 * t276
t280 = r0 ** 2
t281 = r0 ** (0.1e1 / 0.3e1)
t282 = t281 ** 2
t284 = 0.1e1 / t282 / t280
t285 = s0 * t284
t287 = t5 * t8
t288 = 0.1e1 / t241
t291 = lax_cond(0.0e0 < t279, t279, -t279)
t298 = math.log(0.1e1 + t285 * t68 * t287 * t288 / t291 / 0.96e2)
t304 = t84 * t149 * t252
t305 = 0.1e1 / t254
t311 = 0.1e1 + t246 * (0.107924e1 + 0.39640000000000000000e-1 * t246 + 0.12382500000000000000e-1 * t243) / 0.2e1
t312 = t311 ** 2
t319 = t102 * t6
t320 = t87 * t40
t325 = t107 * t5
t326 = t109 * t252
t331 = t170 ** 2
t332 = 0.1e1 / t331
t335 = t119 * t6
t336 = t65 * t40
t341 = -0.37560000000000000000e-1 * t89 * t240 + 0.26347500000000000000e-2 * t319 * t320 * t241 * t240 - 0.47550000000000000000e-3 * t325 * t326 * t254 * t240 + 0.25560000000000000000e-3 * t116 * t332 - 0.21605625000000000000e-5 * t335 * t336 * t241 * t332
t343 = 0.73596627000000000000e-2 * t89 * t240 / t312 - t279 * t341
t346 = t84 * t127 * t22
t350 = t246 * t243 / t311
t354 = t279 ** 2
t357 = 0.1e1 / (0.98556445000000000000e-3 * t346 * t40 * t305 * t350 - 0.2e1 * t354)
t359 = t7 * t170
t360 = t359 ** (0.1e1 / 0.3e1)
t361 = t360 ** 2
t364 = t304 * t305 * t343 * t357 * t285 * t361
t370 = t22 * t40
t380 = (0.61912500000000000000e-2 * t279 * t144 * t350 - 0.39796666666666666666e-1 * t85 * t370 * t305 * t341) * t357 * s0 * t284 * t40 * t361
t383 = s0 ** 2
t385 = t280 ** 2
t392 = t343 * t357 * t383 / t281 / t385 / r0 * t252 * t360 * t359
t396 = (t279 / (0.1e1 + 0.66725e-1 * t298) + 0.24872916666666666666e-2 * t364) / (0.1e1 + t380 / 0.16e2 - t392 / 0.256e3)
t407 = math.log(0.1e1 + 0.32164683177870697974e2 / (0.70594500000000000000e1 * t246 + 0.15494250000000000000e1 * t243 + 0.42077500000000000000e0 * t249 + 0.15629250000000000000e0 * t256))
t415 = -t264 + t188 * (-0.31090e-1 * (0.1e1 + 0.51370000000000000000e-1 * t243) * t407 + t264 - 0.19751789702565206229e-1 * t276) + 0.19751789702565206229e-1 * t188 * t276
t422 = lax_cond(0.0e0 < t415, t415, -t415)
t429 = math.log(0.1e1 + t252 * s0 * t284 * t19 * t210 * t8 * t288 / t422 / 0.192e3)
t450 = lax_cond(t176, p_a_zeta_threshold, t175)
t452 = 0.1e1 / t175
t453 = t452 ** (0.1e1 / 0.3e1)
t455 = t238 * t239 * t453
t458 = math.sqrt(t455)
t461 = t455 ** 0.15e1
t463 = t453 ** 2
t465 = t251 * t253 * t463
t471 = math.log(0.1e1 + 0.16081824322151104822e2 / (0.37978500000000000000e1 * t458 + 0.89690000000000000000e0 * t455 + 0.20477500000000000000e0 * t461 + 0.12323500000000000000e0 * t465))
t473 = 0.62182e-1 * (0.1e1 + 0.53425000000000000000e-1 * t455) * t471
t484 = math.log(0.1e1 + 0.29608574643216675549e2 / (0.51785000000000000000e1 * t458 + 0.90577500000000000000e0 * t455 + 0.11003250000000000000e0 * t461 + 0.12417750000000000000e0 * t465))
t485 = (0.1e1 + 0.27812500000000000000e-1 * t455) * t484
t488 = -t473 + 0.19751789702565206229e-1 * t44 * t485
t489 = r1 ** 2
t490 = r1 ** (0.1e1 / 0.3e1)
t491 = t490 ** 2
t493 = 0.1e1 / t491 / t489
t494 = s2 * t493
t496 = 0.1e1 / t453
t499 = lax_cond(0.0e0 < t488, t488, -t488)
t506 = math.log(0.1e1 + t494 * t68 * t287 * t496 / t499 / 0.96e2)
t511 = 0.1e1 / t463
t517 = 0.1e1 + t458 * (0.107924e1 + 0.39640000000000000000e-1 * t458 + 0.12382500000000000000e-1 * t455) / 0.2e1
t518 = t517 ** 2
t533 = t175 ** 2
t534 = 0.1e1 / t533
t541 = -0.37560000000000000000e-1 * t89 * t452 + 0.26347500000000000000e-2 * t319 * t320 * t453 * t452 - 0.47550000000000000000e-3 * t325 * t326 * t463 * t452 + 0.25560000000000000000e-3 * t116 * t534 - 0.21605625000000000000e-5 * t335 * t336 * t453 * t534
t543 = 0.73596627000000000000e-2 * t89 * t452 / t518 - t488 * t541
t548 = t458 * t455 / t517
t552 = t488 ** 2
t555 = 0.1e1 / (0.98556445000000000000e-3 * t346 * t40 * t511 * t548 - 0.2e1 * t552)
t557 = t7 * t175
t558 = t557 ** (0.1e1 / 0.3e1)
t559 = t558 ** 2
t562 = t304 * t511 * t543 * t555 * t494 * t559
t577 = (0.61912500000000000000e-2 * t488 * t144 * t548 - 0.39796666666666666666e-1 * t85 * t370 * t511 * t541) * t555 * s2 * t493 * t40 * t559
t580 = s2 ** 2
t582 = t489 ** 2
t589 = t543 * t555 * t580 / t490 / t582 / r1 * t252 * t558 * t557
t593 = (t488 / (0.1e1 + 0.66725e-1 * t506) + 0.24872916666666666666e-2 * t562) / (0.1e1 + t577 / 0.16e2 - t589 / 0.256e3)
t604 = math.log(0.1e1 + 0.32164683177870697974e2 / (0.70594500000000000000e1 * t458 + 0.15494250000000000000e1 * t455 + 0.42077500000000000000e0 * t461 + 0.15629250000000000000e0 * t465))
t612 = -t473 + t188 * (-0.31090e-1 * (0.1e1 + 0.51370000000000000000e-1 * t455) * t604 + t473 - 0.19751789702565206229e-1 * t485) + 0.19751789702565206229e-1 * t188 * t485
t619 = lax_cond(0.0e0 < t612, t612, -t612)
t626 = math.log(0.1e1 + t252 * s2 * t493 * t19 * t210 * t8 * t496 / t619 / 0.192e3)
res = t167 + (t174 + t179 - 0.2e1) * t43 * ((t207 / (0.1e1 + 0.66725e-1 * t219) + 0.69644166666666666665e-2 * t141) / (0.1e1 + 0.18750000000000000000e0 * t156 - 0.40468750000000000000e-1 * t163) - t167) - s0 / r0 / tau0 * t236 * (t396 + t188 * ((t415 / (0.1e1 + 0.66725e-1 * t429) + 0.17411041666666666666e-2 * t364) / (0.1e1 + 0.93750000000000000000e-1 * t380 - 0.10117187500000000000e-1 * t392) - t396)) / 0.16e2 - s2 / r1 / tau1 * t450 * (t593 + t188 * ((t612 / (0.1e1 + 0.66725e-1 * t626) + 0.17411041666666666666e-2 * t562) / (0.1e1 + 0.93750000000000000000e-1 * t577 - 0.10117187500000000000e-1 * t589) - t593)) / 0.16e2
