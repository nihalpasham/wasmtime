//! ISLE integration glue code for riscv32 lowering.

// Pull in the ISLE generated code.
#[allow(unused)]
pub mod generated_code;
use generated_code::MInst;

// Types that the generated ISLE code uses via `use super::*`.
use crate::isa;
use crate::isa::riscv32::abi::Riscv32ABICallSite;
use crate::isa::riscv32::lower::args::{
    WritableXReg, XReg,
};
use crate::isa::riscv32::Riscv32Backend;
use crate::machinst::Reg;
use crate::machinst::{isle::*, CallInfo, MachInst};
use crate::machinst::{VCodeConstant, VCodeConstantData};
use crate::{
    ir::{
        immediates::*, types::*, BlockCall, ExternalName, Inst, InstructionData,
        MemFlags, Opcode, TrapCode, Value, ValueList,
    },
    isa::riscv32::inst::*,
    machinst::{ArgPair, InstOutput, IsTailCall},
};
use regalloc2::PReg;
use std::boxed::Box;
use std::vec::Vec;

type BoxCallInfo = Box<CallInfo<ExternalName>>;
type BoxCallIndInfo = Box<CallInfo<Reg>>;
type BoxReturnCallInfo = Box<ReturnCallInfo<ExternalName>>;
type BoxReturnCallIndInfo = Box<ReturnCallInfo<Reg>>;
type BoxExternalName = Box<ExternalName>;
type VecMachLabel = Vec<MachLabel>;
type VecArgPair = Vec<ArgPair>;

pub(crate) struct RV32IsleContext<'a, 'b, I, B>
where
    I: VCodeInst,
    B: LowerBackend,
{
    pub lower_ctx: &'a mut Lower<'b, I>,
    pub backend: &'a B,
    /// Precalucated value for the minimum vector register size. Will be 0 if
    /// vectors are not supported.
    #[allow(dead_code)]
    min_vec_reg_size: u32,
}

impl<'a, 'b> RV32IsleContext<'a, 'b, MInst, Riscv32Backend> {
    fn new(lower_ctx: &'a mut Lower<'b, MInst>, backend: &'a Riscv32Backend) -> Self {
        Self {
            lower_ctx,
            backend,
            min_vec_reg_size: backend.isa_flags.min_vec_reg_size() as u32,
        }
    }
}

impl generated_code::Context for RV32IsleContext<'_, '_, MInst, Riscv32Backend> {
    isle_lower_prelude_methods!();
    isle_prelude_caller_methods!(Riscv32ABICallSite);

    
    fn xreg_new(&mut self, r: Reg) -> XReg {
        XReg::new(r).unwrap()
    }
    fn writable_xreg_new(&mut self, r: WritableReg) -> WritableXReg {
        r.map(|wr| XReg::new(wr).unwrap())
    }
    fn writable_xreg_to_xreg(&mut self, arg0: WritableXReg) -> XReg {
        arg0.to_reg()
    }
    fn writable_xreg_to_writable_reg(&mut self, arg0: WritableXReg) -> WritableReg {
        arg0.map(|xr| xr.to_reg())
    }
    fn xreg_to_reg(&mut self, arg0: XReg) -> Reg {
        *arg0
    }

    fn ty_supported(&mut self, ty: Type) -> Option<Type> {
        // let lane_type = ty.lane_type();
        let supported = match ty {
            // Scalar integers are always supported
            ty if ty.is_int() => true,

            // Otherwise do not match
            _ => false,
        };

        if supported {
            Some(ty)
        } else {
            None
        }
    }

    fn load_ra(&mut self) -> Reg {
        if self.backend.flags.preserve_frame_pointers() {
            let tmp = self.temp_writable_reg(I32);
            self.emit(&MInst::Load {
                rd: tmp,
                op: LoadOP::Lw,
                flags: MemFlags::trusted(),
                from: AMode::FPOffset(8),
            });
            tmp.to_reg()
        } else {
            link_reg()
        }
    }

    fn label_to_br_target(&mut self, label: MachLabel) -> CondBrTarget {
        CondBrTarget::Label(label)
    }

    fn imm12_and(&mut self, imm: Imm12, x: u32) -> Imm12 {
        Imm12::from_i16(imm.as_i16() & (x as i16))
    }

    fn i32_generate_imm(&mut self, imm: i32) -> Option<(Imm20, Imm12)> {
        MInst::generate_imm(imm as u32)
    }

    fn i32_shift_for_lui(&mut self, imm: i32) -> Option<(u32, Imm12)> {
        let trailing = imm.trailing_zeros();
        if trailing < 12 {
            return None;
        }

        let shift = Imm12::from_i16(trailing as i16 - 12);
        let base = (imm as u32) >> trailing;
        Some((base, shift))
    }

    fn i32_shift(&mut self, imm: i32) -> Option<(i32, Imm12)> {
        let trailing = imm.trailing_zeros();
        // We can do without this condition but in this case there is no need to go further
        if trailing == 0 {
            return None;
        }

        let shift = Imm12::from_i16(trailing as i16);
        let base = imm >> trailing;
        Some((base, shift))
    }

    #[inline]
    fn emit(&mut self, arg0: &MInst) -> Unit {
        self.lower_ctx.emit(arg0.clone());
    }
    #[inline]
    fn imm12_from_u32(&mut self, arg0: u32) -> Option<Imm12> {
        Imm12::maybe_from_u32(arg0)
    }
    #[inline]
    fn imm12_from_i32(&mut self, arg0: i32) -> Option<Imm12> {
        Imm12::maybe_from_i32(arg0)
    }
    #[inline]
    fn imm12_is_zero(&mut self, imm: Imm12) -> Option<()> {
        if imm.as_i16() == 0 {
            Some(())
        } else {
            None
        }
    }

    #[inline]
    fn imm20_from_u32(&mut self, arg0: u32) -> Option<Imm20> {
        Imm20::maybe_from_u32(arg0)
    }
    #[inline]
    fn imm20_from_i32(&mut self, arg0: i32) -> Option<Imm20> {
        Imm20::maybe_from_i32(arg0)
    }
    #[inline]
    fn imm20_is_zero(&mut self, imm: Imm20) -> Option<()> {
        if imm.as_i32() == 0 {
            Some(())
        } else {
            None
        }
    }

    #[inline]
    fn imm5_from_u32(&mut self, arg0: u32) -> Option<Imm5> {
        Imm5::maybe_from_i8(i8::try_from(arg0 as i32).ok()?)
    }
    #[inline]
    fn imm5_from_i32(&mut self, arg0: i32) -> Option<Imm5> {
        Imm5::maybe_from_i8(i8::try_from(arg0).ok()?)
    }
    #[inline]
    fn i8_to_imm5(&mut self, arg0: i8) -> Option<Imm5> {
        Imm5::maybe_from_i8(arg0)
    }
    #[inline]
    fn uimm5_bitcast_to_imm5(&mut self, arg0: UImm5) -> Imm5 {
        Imm5::from_bits(arg0.bits() as u8)
    }
    #[inline]
    fn uimm5_from_u8(&mut self, arg0: u8) -> Option<UImm5> {
        UImm5::maybe_from_u8(arg0)
    }
    #[inline]
    fn uimm5_from_u32(&mut self, arg0: u32) -> Option<UImm5> {
        arg0.try_into().ok().and_then(UImm5::maybe_from_u8)
    }
    #[inline]
    fn writable_zero_reg(&mut self) -> WritableReg {
        writable_zero_reg()
    }
    #[inline]
    fn zero_reg(&mut self) -> XReg {
        XReg::new(zero_reg()).unwrap()
    }
    fn is_non_zero_reg(&mut self, reg: XReg) -> Option<()> {
        if reg != self.zero_reg() {
            Some(())
        } else {
            None
        }
    }
    fn is_zero_reg(&mut self, reg: XReg) -> Option<()> {
        if reg == self.zero_reg() {
            Some(())
        } else {
            None
        }
    }
    #[inline]
    fn imm_from_bits(&mut self, val: u32) -> Imm12 {
        Imm12::maybe_from_u32(val).unwrap()
    }
    #[inline]
    fn imm_from_neg_bits(&mut self, val: i32) -> Imm12 {
        Imm12::maybe_from_i32(val).unwrap()
    }

    fn u8_as_i32(&mut self, x: u8) -> i32 {
        x as i32
    }

    fn imm12_const(&mut self, val: i32) -> Imm12 {
        if let Some(res) = Imm12::maybe_from_i32(val) {
            res
        } else {
            panic!("Unable to make an Imm12 value from {val}")
        }
    }
    fn imm12_const_add(&mut self, val: i32, add: i32) -> Imm12 {
        Imm12::maybe_from_i32(val + add).unwrap()
    }
    fn imm12_add(&mut self, val: Imm12, add: i32) -> Option<Imm12> {
        Imm12::maybe_from_i32((i32::from(val.as_i16()) + add).into())
    }

    fn has_v(&mut self) -> bool {
        self.backend.isa_flags.has_v()
    }

    fn has_m(&mut self) -> bool {
        self.backend.isa_flags.has_m()
    }

    fn has_zfa(&mut self) -> bool {
        self.backend.isa_flags.has_zfa()
    }

    fn has_zfh(&mut self) -> bool {
        self.backend.isa_flags.has_zfh()
    }

    fn has_zbkb(&mut self) -> bool {
        self.backend.isa_flags.has_zbkb()
    }

    fn has_zba(&mut self) -> bool {
        self.backend.isa_flags.has_zba()
    }

    fn has_zbb(&mut self) -> bool {
        self.backend.isa_flags.has_zbb()
    }

    fn has_zbc(&mut self) -> bool {
        self.backend.isa_flags.has_zbc()
    }

    fn has_zbs(&mut self) -> bool {
        self.backend.isa_flags.has_zbs()
    }

    fn has_zicond(&mut self) -> bool {
        self.backend.isa_flags.has_zicond()
    }

    fn gen_reg_offset_amode(&mut self, base: Reg, offset: i32) -> AMode {
        AMode::RegOffset(base, offset)
    }

    fn gen_sp_offset_amode(&mut self, offset: i32) -> AMode {
        AMode::SPOffset(offset)
    }

    fn gen_fp_offset_amode(&mut self, offset: i32) -> AMode {
        AMode::FPOffset(offset)
    }

    fn gen_stack_slot_amode(&mut self, ss: StackSlot, offset: i32) -> AMode {
        // Offset from beginning of stackslot area.
        let stack_off = self.lower_ctx.abi().sized_stackslot_offsets()[ss] as i32;
        let sp_off: i32 = stack_off + offset;
        AMode::SlotOffset(sp_off)
    }

    fn gen_const_amode(&mut self, c: VCodeConstant) -> AMode {
        AMode::Const(c)
    }


    fn sinkable_inst(&mut self, val: Value) -> Option<Inst> {
        self.is_sinkable_inst(val)
    }

    fn load_op(&mut self, ty: Type) -> LoadOP {
        LoadOP::from_type(ty)
    }
    fn store_op(&mut self, ty: Type) -> StoreOP {
        StoreOP::from_type(ty)
    }
    fn load_ext_name(&mut self, name: ExternalName, offset: i32) -> Reg {
        let tmp = self.temp_writable_reg(I32);
        self.emit(&MInst::LoadExtName {
            rd: tmp,
            name: Box::new(name),
            offset,
        });
        tmp.to_reg()
    }

    fn gen_stack_addr(&mut self, slot: StackSlot, offset: Offset32) -> Reg {
        let result = self.temp_writable_reg(I32);
        let i = self
            .lower_ctx
            .abi()
            .sized_stackslot_addr(slot, i32::from(offset) as u32, result);
        self.emit(&i);
        result.to_reg()
    }

    fn lower_br_table(&mut self, index: Reg, targets: &[MachLabel]) -> Unit {
        let tmp1 = self.temp_writable_reg(I32);
        let tmp2 = self.temp_writable_reg(I32);
        self.emit(&MInst::BrTable {
            index,
            tmp1,
            tmp2,
            targets: targets.to_vec(),
        });
    }

    fn fp_reg(&mut self) -> PReg {
        px_reg(8)
    }

    fn sp_reg(&mut self) -> PReg {
        px_reg(2)
    }

    #[inline]
    fn int_compare(&mut self, kind: &IntCC, rs1: XReg, rs2: XReg) -> IntegerCompare {
        IntegerCompare {
            kind: *kind,
            rs1: rs1.to_reg(),
            rs2: rs2.to_reg(),
        }
    }

    #[inline]
    fn int_compare_decompose(&mut self, cmp: IntegerCompare) -> (IntCC, XReg, XReg) {
        (cmp.kind, self.xreg_new(cmp.rs1), self.xreg_new(cmp.rs2))
    }

   
}

/// The main entry point for lowering with ISLE.
pub(crate) fn lower(
    lower_ctx: &mut Lower<MInst>,
    backend: &Riscv32Backend,
    inst: Inst,
) -> Option<InstOutput> {
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = RV32IsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower(&mut isle_ctx, inst)
}

/// The main entry point for branch lowering with ISLE.
pub(crate) fn lower_branch(
    lower_ctx: &mut Lower<MInst>,
    backend: &Riscv32Backend,
    branch: Inst,
    targets: &[MachLabel],
) -> Option<()> {
    // TODO: reuse the ISLE context across lowerings so we can reuse its
    // internal heap allocations.
    let mut isle_ctx = RV32IsleContext::new(lower_ctx, backend);
    generated_code::constructor_lower_branch(&mut isle_ctx, branch, targets)
}
