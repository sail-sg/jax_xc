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
t33 = s0 / r0 / tau0 / 0.8e1
t35 = lax_cond(t33 < 0.10e1, t33, 0.10e1)
t36 = t35 ** 2
t37 = t36 * t35
t41 = (0.1e1 + t37) ** 2
t43 = (t36 + 0.3e1 * t37) / t41
t44 = 6 ** (0.1e1 / 0.3e1)
t45 = math.pi ** 2
t46 = t45 ** (0.1e1 / 0.3e1)
t47 = t46 ** 2
t48 = 0.1e1 / t47
t49 = t44 * t48
t50 = r0 ** 2
t51 = r0 ** (0.1e1 / 0.3e1)
t52 = t51 ** 2
t54 = 0.1e1 / t52 / t50
t55 = s0 * t54
t56 = t49 * t55
t58 = t44 ** 2
t61 = t58 / t46 / t45
t62 = s0 ** 2
t63 = t50 ** 2
t71 = (0.1e1 + 0.15045488888888888889e0 * t56 + 0.26899490462262948000e-2 * t61 * t62 / t51 / t63 / r0) ** (0.1e1 / 0.5e1)
t76 = tau0 / t52 / r0
t79 = 0.25633760400000000000e0 * t58 * t47
t86 = t71 ** 2
t102 = (t76 - t55 / 0.8e1) * t44
t105 = 0.5e1 / 0.9e1 * t102 * t48 - 0.1e1
t110 = math.sqrt(0.1e1 + 0.22222222222222222222e0 * t102 * t48 * t105)
t115 = 0.9e1 / 0.20e2 * t105 / t110 + t56 / 0.36e2
t116 = t115 ** 2
t123 = (0.1e1 + 0.5e1 / 0.12e2 * (0.10e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t56) * t44 * t48 * s0 * t54 + 0.292e3 / 0.405e3 * t116 - 0.146e3 / 0.135e3 * t115 * t35 * (0.1e1 - t35)) ** (0.1e1 / 0.10e2)
t129 = lax_cond(r0 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t26 * t27 * (t43 * (0.1e1 / t71 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t56 - 0.5e1 / 0.9e1 * (0.14554132000000000000e0 * t76 + t79 + 0.11867481666666666667e-1 * t55) * t44 * t48) / t86) + (0.1e1 - t43) * t123))
t131 = lax_cond(t10, t15, -t17)
t132 = lax_cond(t14, t11, t131)
t133 = 0.1e1 + t132
t135 = t133 ** (0.1e1 / 0.3e1)
t137 = lax_cond(t133 <= p_a_zeta_threshold, t23, t135 * t133)
t143 = s2 / r1 / tau1 / 0.8e1
t145 = lax_cond(t143 < 0.10e1, t143, 0.10e1)
t146 = t145 ** 2
t147 = t146 * t145
t151 = (0.1e1 + t147) ** 2
t153 = (t146 + 0.3e1 * t147) / t151
t154 = r1 ** 2
t155 = r1 ** (0.1e1 / 0.3e1)
t156 = t155 ** 2
t158 = 0.1e1 / t156 / t154
t159 = s2 * t158
t160 = t49 * t159
t162 = s2 ** 2
t163 = t154 ** 2
t171 = (0.1e1 + 0.15045488888888888889e0 * t160 + 0.26899490462262948000e-2 * t61 * t162 / t155 / t163 / r1) ** (0.1e1 / 0.5e1)
t176 = tau1 / t156 / r1
t184 = t171 ** 2
t200 = (t176 - t159 / 0.8e1) * t44
t203 = 0.5e1 / 0.9e1 * t200 * t48 - 0.1e1
t208 = math.sqrt(0.1e1 + 0.22222222222222222222e0 * t200 * t48 * t203)
t213 = 0.9e1 / 0.20e2 * t203 / t208 + t160 / 0.36e2
t214 = t213 ** 2
t221 = (0.1e1 + 0.5e1 / 0.12e2 * (0.10e2 / 0.81e2 + 0.25e2 / 0.8748e4 * t160) * t44 * t48 * s2 * t158 + 0.292e3 / 0.405e3 * t214 - 0.146e3 / 0.135e3 * t213 * t145 * (0.1e1 - t145)) ** (0.1e1 / 0.10e2)
t227 = lax_cond(r1 <= p_a_dens_threshold, 0, -0.3e1 / 0.8e1 * t5 * t137 * t27 * (t153 * (0.1e1 / t171 + 0.7e1 / 0.9e1 * (0.1e1 + 0.63943327777777777778e-1 * t160 - 0.5e1 / 0.9e1 * (0.14554132000000000000e0 * t176 + t79 + 0.11867481666666666667e-1 * t159) * t44 * t48) / t184) + (0.1e1 - t153) * t221))
res = t129 + t227
