;;! target = "aarch64"
;;! test = "winch"

(module
    (func (result i64)
	(i64.const -1)
	(i64.const -1)
	(i64.div_u)
    )
)

;; wasm[0]::function[0]:
;;       stp     x29, x30, [sp, #-0x10]!
;;       mov     x29, sp
;;       mov     x28, sp
;;       mov     x9, x0
;;       sub     sp, sp, #0x10
;;       mov     x28, sp
;;       stur    x0, [x28, #8]
;;       stur    x1, [x28]
;;       mov     x16, #-1
;;       mov     x0, x16
;;       mov     x16, #-1
;;       mov     x1, x16
;;       cbz     x0, #0x4c
;;   34: udiv    x1, x1, x0
;;       mov     x0, x1
;;       add     sp, sp, #0x10
;;       mov     x28, sp
;;       ldp     x29, x30, [sp], #0x10
;;       ret
;;   4c: .byte   0x1f, 0xc1, 0x00, 0x00
