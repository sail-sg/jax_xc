t2 = 3 ** (0.1e1 / 0.3e1)
t3 = t2 ** 2
t4 = math.pi ** (0.1e1 / 0.3e1)
t7 = t3 / t4 / math.pi
t8 = r0 + r1
t9 = 0.1e1 / t8
t12 = 0.2e1 * r0 * t9 <= p_a_zeta_threshold
t13 = p_a_zeta_threshold - 0.1e1
t16 = 0.2e1 * r1 * t9 <= p_a_zeta_threshold
t17 = -t13
t19 = (r0 - r1) * t9
t20 = lax_cond(t16, t17, t19)
t21 = lax_cond(t12, t13, t20)
t22 = 0.1e1 + t21
t24 = p_a_zeta_threshold ** (0.1e1 / 0.3e1)
t25 = t24 * p_a_zeta_threshold
t26 = t22 ** (0.1e1 / 0.3e1)
t28 = lax_cond(t22 <= p_a_zeta_threshold, t25, t26 * t22)
t29 = t8 ** (0.1e1 / 0.3e1)
t32 = math.pi ** 2
t33 = math.sqrt(s0)
t34 = r0 ** (0.1e1 / 0.3e1)
t37 = t33 / t34 / r0
t38 = 4 ** (0.1e1 / 0.3e1)
t39 = math.sqrt(0.3e1)
t41 = t39 * t32 * math.pi
t42 = t41 ** (0.1e1 / 0.3e1)
t43 = t38 * t42
t45 = 3 ** (0.1e1 / 0.4e1)
t46 = math.sqrt(0.2e1)
t48 = math.sqrt(math.pi)
t51 = t45 * t46 / t48 / math.pi
t52 = t43 - 0.1e-9
t54 = lax_cond(t52 < t37, t52, t37)
t55 = t54 ** 2
t56 = 0.4e1 * t41
t57 = t32 ** 2
t58 = t57 * t32
t59 = 0.48e2 * t58
t60 = t55 ** 2
t63 = math.sqrt(-t60 * t55 + t59)
t64 = t56 + t63
t65 = t64 ** (0.1e1 / 0.3e1)
t66 = t65 ** 2
t68 = math.sqrt(t55 + t66)
t70 = t64 ** (0.1e1 / 0.6e1)
t75 = math.asinh(t51 * t54 * t68 / t70 / 0.4e1)
t76 = 0.1e1 / math.pi
t77 = t43 + 0.1e-9
t79 = lax_cond(t77 < t37, t37, t77)
t80 = t79 ** 2
t83 = 0.1e1 / t58
t84 = t80 ** 2
t89 = math.sqrt(0.3e1 * t83 * t84 * t80 - 0.144e3)
t91 = math.atan(t89 / 0.12e2)
t93 = math.cos(t91 / 0.3e1)
t96 = math.sqrt(t80 * t79 * t39 * t76 * t93)
t99 = math.asinh(t76 * t96 / 0.2e1)
t100 = lax_cond(t37 < t43, t75, t99)
t102 = math.exp(-0.2e1 * t100)
t104 = math.log(0.1e1 + t102)
t107 = my_dilog(-t102)
t112 = 0.1e1 / math.cosh(t100)
t113 = t112 ** (0.1e1 / 0.3e1)
t114 = t113 ** 2
t116 = t76 ** (0.1e1 / 0.3e1)
t117 = 0.1e1 / t116
t123 = lax_cond(r0 <= p_a_dens_threshold, 0, -t7 * t28 * t29 * (-0.12e2 * t100 * t104 + 0.12e2 * t107 + t32) / t100 / t114 * t117 * t38 / 0.24e2)
t125 = lax_cond(t12, t17, -t19)
t126 = lax_cond(t16, t13, t125)
t127 = 0.1e1 + t126
t129 = t127 ** (0.1e1 / 0.3e1)
t131 = lax_cond(t127 <= p_a_zeta_threshold, t25, t129 * t127)
t134 = math.sqrt(s2)
t135 = r1 ** (0.1e1 / 0.3e1)
t138 = t134 / t135 / r1
t141 = lax_cond(t52 < t138, t52, t138)
t142 = t141 ** 2
t143 = t142 ** 2
t146 = math.sqrt(-t143 * t142 + t59)
t147 = t56 + t146
t148 = t147 ** (0.1e1 / 0.3e1)
t149 = t148 ** 2
t151 = math.sqrt(t142 + t149)
t153 = t147 ** (0.1e1 / 0.6e1)
t158 = math.asinh(t51 * t141 * t151 / t153 / 0.4e1)
t160 = lax_cond(t77 < t138, t138, t77)
t161 = t160 ** 2
t164 = t161 ** 2
t169 = math.sqrt(0.3e1 * t83 * t164 * t161 - 0.144e3)
t171 = math.atan(t169 / 0.12e2)
t173 = math.cos(t171 / 0.3e1)
t176 = math.sqrt(t161 * t160 * t39 * t76 * t173)
t179 = math.asinh(t76 * t176 / 0.2e1)
t180 = lax_cond(t138 < t43, t158, t179)
t182 = math.exp(-0.2e1 * t180)
t184 = math.log(0.1e1 + t182)
t187 = my_dilog(-t182)
t192 = 0.1e1 / math.cosh(t180)
t193 = t192 ** (0.1e1 / 0.3e1)
t194 = t193 ** 2
t201 = lax_cond(r1 <= p_a_dens_threshold, 0, -t7 * t131 * t29 * (-0.12e2 * t180 * t184 + 0.12e2 * t187 + t32) / t180 / t194 * t117 * t38 / 0.24e2)
res = t123 + t201
