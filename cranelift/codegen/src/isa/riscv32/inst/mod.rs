//! This module defines RV32-specific machine instruction types.
#![allow(warnings)] // temporary flag

use crate::binemit::{Addend, CodeOffset, Reloc};
pub use crate::ir::condcodes::IntCC;
use crate::ir::types::{self, F16, F32, F64, I128, I16, I32, I64, I8, I8X16};

pub use crate::ir::{ExternalName, MemFlags, Type};
use crate::isa::{CallConv, FunctionAlignment};
use crate::machinst::*;
use crate::{settings, CodegenError, CodegenResult};

pub use crate::ir::condcodes::FloatCC;

use alloc::vec::Vec;
use regalloc2::RegClass;
use smallvec::{smallvec, SmallVec};
use std::boxed::Box;
use std::fmt::Write;
use std::string::{String, ToString};

pub mod regs;
pub use self::regs::*;
pub mod imms;
pub use self::imms::*;
pub mod args;
pub use self::args::*;
pub mod emit;
pub use self::emit::*;
pub mod encode;
pub use self::encode::*;
pub mod unwind;

use crate::isa::riscv32::abi::Riscv32MachineDeps;

#[cfg(test)]
mod emit_tests;

use std::fmt::{Display, Formatter};

pub(crate) type VecU8 = Vec<u8>;

//=============================================================================
// Instructions (top level): definition

use crate::isa::riscv32::lower::isle::generated_code::MInst;
pub use crate::isa::riscv32::lower::isle::generated_code::{
    AluOPRRI, AluOPRRR, CsrImmOP, CsrRegOP, LoadOP, MInst as Inst, StoreOP, CSR,
};

/// Additional information for `return_call[_ind]` instructions, left out of
/// line to lower the size of the `Inst` enum.
#[derive(Clone, Debug)]
pub struct ReturnCallInfo<T> {
    pub dest: T,
    pub uses: CallArgList,
    pub new_stack_arg_size: u32,
}

/// A conditional branch target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CondBrTarget {
    /// An unresolved reference to a Label, as passed into
    /// `lower_branch_group()`.
    Label(MachLabel),
    /// No jump; fall through to the next instruction.
    Fallthrough,
}

impl CondBrTarget {
    /// Return the target's label, if it is a label-based target.
    pub(crate) fn as_label(self) -> Option<MachLabel> {
        match self {
            CondBrTarget::Label(l) => Some(l),
            _ => None,
        }
    }

    pub(crate) fn is_fallthrouh(&self) -> bool {
        self == &CondBrTarget::Fallthrough
    }
}

impl Display for CondBrTarget {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            CondBrTarget::Label(l) => write!(f, "{}", l.to_string()),
            CondBrTarget::Fallthrough => write!(f, "0"),
        }
    }
}

pub(crate) fn enc_auipc(rd: Writable<Reg>, imm: Imm20) -> u32 {
    let x = 0b0010111 | reg_to_gpr_num(rd.to_reg()) << 7 | imm.bits() << 12;
    x
}

pub(crate) fn enc_jalr(rd: Writable<Reg>, base: Reg, offset: Imm12) -> u32 {
    let x = 0b1100111
        | reg_to_gpr_num(rd.to_reg()) << 7
        | 0b000 << 12
        | reg_to_gpr_num(base) << 15
        | offset.bits() << 20;
    x
}

/// rd and src must have the same length.
pub(crate) fn gen_moves(rd: &[Writable<Reg>], src: &[Reg]) -> SmallInstVec<Inst> {
    assert!(rd.len() == src.len());
    assert!(rd.len() > 0);
    let mut insts = SmallInstVec::new();
    for (dst, src) in rd.iter().zip(src.iter()) {
        let ty = Inst::canonical_type_for_rc(dst.to_reg().class());
        insts.push(Inst::gen_move(*dst, *src, ty));
    }
    insts
}

impl Inst {
    /// RISC-V can have multiple instruction sizes. 2 bytes for compressed
    /// instructions, 4 for regular instructions, 6 and 8 byte instructions
    /// are also being considered.
    const UNCOMPRESSED_INSTRUCTION_SIZE: i32 = 4;

    #[inline]
    pub(crate) fn load_imm12(rd: Writable<Reg>, imm: Imm12) -> Inst {
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Addi,
            rd,
            rs: zero_reg(),
            imm12: imm,
        }
    }

    /// Immediates can be loaded using lui and addi instructions.
    fn load_const_imm(rd: Writable<Reg>, value: u32) -> Option<SmallInstVec<Inst>> {
        Inst::generate_imm(value).map(|(imm20, imm12)| {
            let mut insts = SmallVec::new();

            let imm20_is_zero = imm20.as_i32() == 0;
            let imm12_is_zero = imm12.as_i16() == 0;

            let rs = if !imm20_is_zero {
                insts.push(Inst::Lui { rd, imm: imm20 });
                rd.to_reg()
            } else {
                zero_reg()
            };

            // We also need to emit the addi if the value is 0, otherwise we just
            // won't produce any instructions.
            if !imm12_is_zero || (imm20_is_zero && imm12_is_zero) {
                insts.push(Inst::AluRRImm12 {
                    alu_op: AluOPRRI::Addi,
                    rd,
                    rs,
                    imm12,
                })
            }

            insts
        })
    }

    pub(crate) fn load_constant_u32(rd: Writable<Reg>, value: u32) -> SmallInstVec<Inst> {
        let insts = Inst::load_const_imm(rd, value);
        insts.unwrap_or_else(|| {
            smallvec![Inst::LoadInlineConst {
                rd,
                ty: I32,
                imm: value
            }]
        })
    }

    pub(crate) fn construct_auipc_and_jalr(
        link: Option<Writable<Reg>>,
        tmp: Writable<Reg>,
        offset: i32,
    ) -> [Inst; 2] {
        Inst::generate_imm(offset as u32)
            .map(|(imm20, imm12)| {
                let a = Inst::Auipc {
                    rd: tmp,
                    imm: imm20,
                };
                let b = Inst::Jalr {
                    rd: link.unwrap_or(writable_zero_reg()),
                    base: tmp.to_reg(),
                    offset: imm12,
                };
                [a, b]
            })
            .expect("code range is too big.")
    }

    /// Generic constructor for a load (zero-extending where appropriate).
    pub fn gen_load(into_reg: Writable<Reg>, mem: AMode, ty: Type, flags: MemFlags) -> Inst {
        Inst::Load {
            rd: into_reg,
            op: LoadOP::from_type(ty),
            from: mem,
            flags,
        }
    }

    /// Generic constructor for a store.
    pub fn gen_store(mem: AMode, from_reg: Reg, ty: Type, flags: MemFlags) -> Inst {
        Inst::Store {
            src: from_reg,
            op: StoreOP::from_type(ty),
            to: mem,
            flags,
        }
    }
}

fn riscv32_get_operands(inst: &mut Inst, collector: &mut impl OperandVisitor) {
    match inst {
        Inst::Nop0 | Inst::Nop4 => {}
        Inst::BrTable {
            index, tmp1, tmp2, ..
        } => {
            collector.reg_use(index);
            collector.reg_early_def(tmp1);
            collector.reg_early_def(tmp2);
        }
        Inst::Auipc { rd, .. } => collector.reg_def(rd),
        Inst::Lui { rd, .. } => collector.reg_def(rd),
        Inst::LoadInlineConst { rd, .. } => collector.reg_def(rd),

        Inst::AluRRR { rd, rs1, rs2, .. } => {
            collector.reg_use(rs1);
            collector.reg_use(rs2);
            collector.reg_def(rd);
        }

        Inst::AluRRImm12 { rd, rs, .. } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::CsrReg { rd, rs, .. } => {
            collector.reg_use(rs);
            collector.reg_def(rd);
        }
        Inst::CsrImm { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::Load { rd, from, .. } => {
            from.get_operands(collector);
            collector.reg_def(rd);
        }
        Inst::Store { to, src, .. } => {
            to.get_operands(collector);
            collector.reg_use(src);
        }
        Inst::Args { args } => {
            for ArgPair { vreg, preg } in args {
                collector.reg_fixed_def(vreg, *preg);
            }
        }
        Inst::Rets { rets } => {
            for RetPair { vreg, preg } in rets {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::Ret { .. } => {}
        Inst::Extend { rd, rn, .. } => {
            collector.reg_use(rn);
            collector.reg_def(rd);
        }
        Inst::Call { info, .. } => {
            let CallInfo { uses, defs, .. } = &mut **info;
            for CallArgPair { vreg, preg } in uses {
                collector.reg_fixed_use(vreg, *preg);
            }
            for CallRetPair { vreg, preg } in defs {
                collector.reg_fixed_def(vreg, *preg);
            }
            collector.reg_clobbers(info.clobbers);
        }
        Inst::CallInd { info } => {
            let CallInfo {
                dest, uses, defs, ..
            } = &mut **info;
            collector.reg_use(dest);
            for CallArgPair { vreg, preg } in uses {
                collector.reg_fixed_use(vreg, *preg);
            }
            for CallRetPair { vreg, preg } in defs {
                collector.reg_fixed_def(vreg, *preg);
            }
            collector.reg_clobbers(info.clobbers);
        }
        Inst::ReturnCall { info } => {
            for CallArgPair { vreg, preg } in &mut info.uses {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::ReturnCallInd { info } => {
            // TODO(https://github.com/bytecodealliance/regalloc2/issues/145):
            // This shouldn't be a fixed register constraint.
            collector.reg_fixed_use(&mut info.dest, x_reg(5));

            for CallArgPair { vreg, preg } in &mut info.uses {
                collector.reg_fixed_use(vreg, *preg);
            }
        }
        Inst::Jal { .. } => {
            // JAL technically has a rd register, but we currently always
            // hardcode it to x0.
        }
        Inst::CondBr {
            kind: IntegerCompare { rs1, rs2, .. },
            ..
        } => {
            collector.reg_use(rs1);
            collector.reg_use(rs2);
        }
        Inst::LoadExtName { rd, .. } => {
            collector.reg_def(rd);
        }
        Inst::ElfTlsGetAddr { rd, .. } => {
            // x10 is a0 which is both the first argument and the first return value.
            collector.reg_fixed_def(rd, a0());
            let mut clobbers = Riscv32MachineDeps::get_regs_clobbered_by_call(CallConv::SystemV);
            clobbers.remove(px_reg(10));
            collector.reg_clobbers(clobbers);
        }
        Inst::LoadAddr { rd, mem } => {
            mem.get_operands(collector);
            collector.reg_early_def(rd);
        }

        Inst::Mov { rd, rm, .. } => {
            collector.reg_use(rm);
            collector.reg_def(rd);
        }

        Inst::Fence { .. } => {}
        Inst::EBreak => {}
        Inst::Udf { .. } => {}

        Inst::Jalr { rd, base, .. } => {
            collector.reg_use(base);
            collector.reg_def(rd);
        }
        Inst::Select {
            dst,
            condition: IntegerCompare { rs1, rs2, .. },
            x,
            y,
            ..
        } => {
            // Mark the condition registers as late use so that they don't overlap with the destination
            // register. We may potentially write to the destination register before evaluating the
            // condition.
            collector.reg_late_use(rs1);
            collector.reg_late_use(rs2);

            for reg in x.regs_mut() {
                collector.reg_use(reg);
            }
            for reg in y.regs_mut() {
                collector.reg_use(reg);
            }

            // If there's more than one destination register then use
            // `reg_early_def` to prevent destination registers from overlapping
            // with any operands. This ensures that the lowering doesn't have to
            // deal with a situation such as when the input registers need to be
            // swapped when moved to the destination.
            //
            // When there's only one destination register though don't use an
            // early def because once the register is written no other inputs
            // are read so it's ok for the destination to overlap the sources.
            // The condition registers are already marked as late use so they
            // won't overlap with the destination.
            match dst.regs_mut() {
                [reg] => collector.reg_def(reg),
                regs => {
                    for d in regs {
                        collector.reg_early_def(d);
                    }
                }
            }
        }
        Inst::Popcnt {
            sum, step, rs, tmp, ..
        } => {
            collector.reg_use(rs);
            collector.reg_early_def(tmp);
            collector.reg_early_def(step);
            collector.reg_early_def(sum);
        }
        Inst::Cltz {
            sum, step, tmp, rs, ..
        } => {
            collector.reg_use(rs);
            collector.reg_early_def(tmp);
            collector.reg_early_def(step);
            collector.reg_early_def(sum);
        }
        Inst::Brev8 {
            rs,
            rd,
            step,
            tmp,
            tmp2,
            ..
        } => {
            collector.reg_use(rs);
            collector.reg_early_def(step);
            collector.reg_early_def(tmp);
            collector.reg_early_def(tmp2);
            collector.reg_early_def(rd);
        }
        Inst::StackProbeLoop { .. } => {
            // StackProbeLoop has a tmp register and StackProbeLoop used at gen_prologue.
            // t3 will do the job. (t3 is caller-save register and not used directly by compiler like writable_spilltmp_reg)
            // gen_prologue is called at emit stage.
            // no need let reg alloc know.
        }
        Inst::RawData { .. } => {}
        Inst::TrapIf { .. } | Inst::DummyUse { .. } | Inst::Unwind { .. } => {}
    }
}

impl MachInst for Inst {
    type LabelUse = LabelUse;
    type ABIMachineSpec = Riscv32MachineDeps;

    // https://github.com/riscv/riscv-isa-manual/issues/850
    // all zero will cause invalid opcode.
    const TRAP_OPCODE: &'static [u8] = &[0; 4];

    fn gen_dummy_use(reg: Reg) -> Self {
        Inst::DummyUse { reg }
    }

    fn canonical_type_for_rc(rc: RegClass) -> Type {
        match rc {
            regalloc2::RegClass::Int => I64,
            regalloc2::RegClass::Float => F64,
            regalloc2::RegClass::Vector => I8X16,
        }
    }

    fn is_safepoint(&self) -> bool {
        match self {
            Inst::Call { .. } | Inst::CallInd { .. } => true,
            _ => false,
        }
    }

    fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        riscv32_get_operands(self, collector);
    }

    fn is_move(&self) -> Option<(Writable<Reg>, Reg)> {
        match self {
            _ => None,
        }
    }

    fn is_included_in_clobbers(&self) -> bool {
        match self {
            _ => true,
        }
    }

    fn is_trap(&self) -> bool {
        match self {
            Self::Udf { .. } => true,
            _ => false,
        }
    }

    fn is_args(&self) -> bool {
        match self {
            _ => false,
        }
    }

    fn is_term(&self) -> MachTerminator {
        match self {
            &Inst::Jal { .. } => MachTerminator::Uncond,

            &Inst::Jalr { .. } => MachTerminator::Uncond,

            &Inst::ReturnCall { .. } | &Inst::ReturnCallInd { .. } => MachTerminator::RetCall,
            _ => MachTerminator::None,
        }
    }

    fn is_mem_access(&self) -> bool {
        panic!("TODO FILL ME OUT")
    }

    fn gen_move(_to_reg: Writable<Reg>, _from_reg: Reg, _ty: Type) -> Inst {
        unimplemented!()
    }

    fn gen_nop(preferred_size: usize) -> Inst {
        if preferred_size == 0 {
            return Inst::Nop0;
        }
        // We can't give a NOP (or any insn) < 4 bytes.
        assert!(preferred_size >= 4);
        Inst::Nop4
    }

    fn rc_for_type(ty: Type) -> CodegenResult<(&'static [RegClass], &'static [Type])> {
        match ty {
            I8 => Ok((&[RegClass::Int], &[I8])),
            I16 => Ok((&[RegClass::Int], &[I16])),
            I32 => Ok((&[RegClass::Int], &[I32])),
            I64 => Ok((&[RegClass::Int], &[I64])),
            F16 => Ok((&[RegClass::Float], &[F16])),
            F32 => Ok((&[RegClass::Float], &[F32])),
            F64 => Ok((&[RegClass::Float], &[F64])),
            I128 => Ok((&[RegClass::Int, RegClass::Int], &[I64, I64])),
            _ if ty.is_vector() => {
                debug_assert!(ty.bits() <= 512);

                // Here we only need to return a SIMD type with the same size as `ty`.
                // We use these types for spills and reloads, so prefer types with lanes <= 31
                // since that fits in the immediate field of `vsetivli`.
                const SIMD_TYPES: [[Type; 1]; 6] = [
                    [types::I8X2],
                    [types::I8X4],
                    [types::I8X8],
                    [types::I8X16],
                    [types::I16X16],
                    [types::I32X16],
                ];
                let idx = (ty.bytes().ilog2() - 1) as usize;
                let ty = &SIMD_TYPES[idx][..];

                Ok((&[RegClass::Vector], ty))
            }
            _ => Err(CodegenError::Unsupported(format!(
                "Unexpected SSA-value type: {ty}"
            ))),
        }
    }

    fn gen_jump(target: MachLabel) -> Inst {
        Inst::Jal { label: target }
    }

    fn worst_case_size() -> CodeOffset {
        // Our worst case size is determined by the riscv32_worst_case_instruction_size test
        84
    }

    fn ref_type_regclass(_settings: &settings::Flags) -> RegClass {
        RegClass::Int
    }

    fn function_alignment() -> FunctionAlignment {
        FunctionAlignment {
            minimum: 2,
            preferred: 4,
        }
    }
}

//=============================================================================
// Pretty-printing of instructions.
pub fn reg_name(reg: Reg) -> String {
    match reg.to_real_reg() {
        Some(real) => match real.class() {
            RegClass::Int => match real.hw_enc() {
                0 => "zero".into(),
                1 => "ra".into(),
                2 => "sp".into(),
                3 => "gp".into(),
                4 => "tp".into(),
                5..=7 => format!("t{}", real.hw_enc() - 5),
                8 => "fp".into(),
                9 => "s1".into(),
                10..=17 => format!("a{}", real.hw_enc() - 10),
                18..=27 => format!("s{}", real.hw_enc() - 16),
                28..=31 => format!("t{}", real.hw_enc() - 25),
                _ => unreachable!(),
            },
            RegClass::Float => match real.hw_enc() {
                0..=7 => format!("ft{}", real.hw_enc() - 0),
                8..=9 => format!("fs{}", real.hw_enc() - 8),
                10..=17 => format!("fa{}", real.hw_enc() - 10),
                18..=27 => format!("fs{}", real.hw_enc() - 16),
                28..=31 => format!("ft{}", real.hw_enc() - 20),
                _ => unreachable!(),
            },
            RegClass::Vector => format!("v{}", real.hw_enc()),
        },
        None => {
            format!("{reg:?}")
        }
    }
}

impl Inst {
    fn print_with_state(&self, _state: &mut EmitState) -> String {
        let format_reg = |reg: Reg| -> String { reg_name(reg) };

        let format_regs = |regs: &[Reg]| -> String {
            let mut x = if regs.len() > 1 {
                String::from("[")
            } else {
                String::default()
            };
            regs.iter().for_each(|i| {
                x.push_str(format_reg(*i).as_str());
                if *i != *regs.last().unwrap() {
                    x.push_str(",");
                }
            });
            if regs.len() > 1 {
                x.push_str("]");
            }
            x
        };
        let format_labels = |labels: &[MachLabel]| -> String {
            if labels.len() == 0 {
                return String::from("[_]");
            }
            let mut x = String::from("[");
            labels.iter().for_each(|l| {
                x.push_str(
                    format!(
                        "{:?}{}",
                        l,
                        if l != labels.last().unwrap() { "," } else { "" },
                    )
                    .as_str(),
                );
            });
            x.push_str("]");
            x
        };

        match self {
            &Inst::Nop0 => {
                format!("##zero length nop")
            }
            &Inst::Nop4 => {
                format!("##fixed 4-size nop")
            }
            &Inst::StackProbeLoop {
                guard_size,
                probe_count,
                tmp,
            } => {
                let tmp = format_reg(tmp.to_reg());
                format!(
                    "inline_stack_probe##guard_size={guard_size} probe_count={probe_count} tmp={tmp}"
                )
            }
            &Inst::DummyUse { reg } => {
                let reg = format_reg(reg);
                format!("dummy_use {reg}")
            }

            &Inst::Unwind { ref inst } => {
                format!("unwind {inst:?}")
            }
            &Inst::Brev8 {
                rs,
                ty,
                step,
                tmp,
                tmp2,
                rd,
            } => {
                let rs = format_reg(rs);
                let step = format_reg(step.to_reg());
                let tmp = format_reg(tmp.to_reg());
                let tmp2 = format_reg(tmp2.to_reg());
                let rd = format_reg(rd.to_reg());
                format!("brev8 {rd},{rs}##tmp={tmp} tmp2={tmp2} step={step} ty={ty}")
            }
            &Inst::Popcnt {
                sum,
                step,
                rs,
                tmp,
                ty,
            } => {
                let rs = format_reg(rs);
                let tmp = format_reg(tmp.to_reg());
                let step = format_reg(step.to_reg());
                let sum = format_reg(sum.to_reg());
                format!("popcnt {sum},{rs}##ty={ty} tmp={tmp} step={step}")
            }
            &Inst::Cltz {
                sum,
                step,
                rs,
                tmp,
                ty,
                leading,
            } => {
                let rs = format_reg(rs);
                let tmp = format_reg(tmp.to_reg());
                let step = format_reg(step.to_reg());
                let sum = format_reg(sum.to_reg());
                format!(
                    "{} {},{}##ty={} tmp={} step={}",
                    if leading { "clz" } else { "ctz" },
                    sum,
                    rs,
                    ty,
                    tmp,
                    step
                )
            }
            &Inst::RawData { ref data } => match data.len() {
                4 => {
                    let mut bytes = [0; 4];
                    for i in 0..bytes.len() {
                        bytes[i] = data[i];
                    }
                    format!(".4byte 0x{:x}", u32::from_le_bytes(bytes))
                }
                8 => {
                    let mut bytes = [0; 8];
                    for i in 0..bytes.len() {
                        bytes[i] = data[i];
                    }
                    format!(".8byte 0x{:x}", u64::from_le_bytes(bytes))
                }
                _ => {
                    format!(".data {data:?}")
                }
            },
            &Inst::Auipc { rd, imm } => {
                format!("{} {},{}", "auipc", format_reg(rd.to_reg()), imm.as_i32(),)
            }
            &Inst::Jalr { rd, base, offset } => {
                let base = format_reg(base);
                let rd = format_reg(rd.to_reg());
                format!("{} {},{}({})", "jalr", rd, offset.as_i16(), base)
            }
            &Inst::Lui { rd, ref imm } => {
                format!("{} {},{}", "lui", format_reg(rd.to_reg()), imm.as_i32())
            }
            &Inst::LoadInlineConst { rd, imm, .. } => {
                let rd = format_reg(rd.to_reg());
                let mut buf = String::new();
                write!(&mut buf, "auipc {rd},0; ").unwrap();
                write!(&mut buf, "ld {rd},12({rd}); ").unwrap();
                write!(&mut buf, "j {}; ", Inst::UNCOMPRESSED_INSTRUCTION_SIZE + 8).unwrap();
                write!(&mut buf, ".8byte 0x{imm:x}").unwrap();
                buf
            }

            &Inst::AluRRR {
                alu_op,
                rd,
                rs1,
                rs2,
            } => {
                let rs1_s = format_reg(rs1);
                let rs2_s = format_reg(rs2);
                let rd_s = format_reg(rd.to_reg());
                match alu_op {
                    _ => {
                        format!("{} {},{},{}", alu_op.op_name(), rd_s, rs1_s, rs2_s)
                    }
                }
            }

            &Inst::AluRRImm12 {
                alu_op,
                rd,
                rs,
                ref imm12,
            } => {
                let rs_s = format_reg(rs);
                let rd = format_reg(rd.to_reg());

                // Some of these special cases are better known as
                // their pseudo-instruction version, so prefer printing those.
                match (alu_op, rs, imm12) {
                    (AluOPRRI::Addi, rs, _) if rs == zero_reg() => {
                        return format!("li {},{}", rd, imm12.as_i16());
                    }

                    (AluOPRRI::Xori, _, imm12) if imm12.as_i16() == -1 => {
                        return format!("not {rd},{rs_s}");
                    }
                    (AluOPRRI::SltiU, _, imm12) if imm12.as_i16() == 1 => {
                        return format!("seqz {rd},{rs_s}");
                    }
                    (alu_op, _, _) if alu_op.option_funct12().is_some() => {
                        format!("{} {},{}", alu_op.op_name(), rd, rs_s)
                    }
                    (alu_op, _, imm12) => {
                        format!("{} {},{},{}", alu_op.op_name(), rd, rs_s, imm12.as_i16())
                    }
                }
            }
            &Inst::CsrReg { op, rd, rs, csr } => {
                let rs_s = format_reg(rs);
                let rd_s = format_reg(rd.to_reg());

                match (op, csr, rd) {
                    (CsrRegOP::CsrRW, CSR::Frm, rd) if rd.to_reg() == zero_reg() => {
                        format!("fsrm {rs_s}")
                    }
                    _ => {
                        format!("{op} {rd_s},{csr},{rs_s}")
                    }
                }
            }
            &Inst::CsrImm { op, rd, csr, imm } => {
                let rd_s = format_reg(rd.to_reg());

                match (op, csr, rd) {
                    (CsrImmOP::CsrRWI, CSR::Frm, rd) if rd.to_reg() != zero_reg() => {
                        format!("fsrmi {rd_s},{imm}")
                    }
                    _ => {
                        format!("{op} {rd_s},{csr},{imm}")
                    }
                }
            }
            &Inst::BrTable {
                index,
                tmp1,
                tmp2,
                ref targets,
            } => {
                format!(
                    "{} {},{}##tmp1={},tmp2={}",
                    "br_table",
                    format_reg(index),
                    format_labels(&targets[..]),
                    format_reg(tmp1.to_reg()),
                    format_reg(tmp2.to_reg()),
                )
            }
            &Inst::Load {
                rd,
                op,
                from,
                flags: _flags,
            } => {
                let base = from.to_string();
                let rd = format_reg(rd.to_reg());
                format!("{} {},{}", op.op_name(), rd, base,)
            }
            &Inst::Store {
                to,
                src,
                op,
                flags: _flags,
            } => {
                let base = to.to_string();
                let src = format_reg(src);
                format!("{} {},{}", op.op_name(), src, base,)
            }
            &Inst::Args { ref args } => {
                let mut s = "args".to_string();
                for arg in args {
                    let preg = format_reg(arg.preg);
                    let def = format_reg(arg.vreg.to_reg());
                    write!(&mut s, " {def}={preg}").unwrap();
                }
                s
            }
            &Inst::Rets { ref rets } => {
                let mut s = "rets".to_string();
                for ret in rets {
                    let preg = format_reg(ret.preg);
                    let vreg = format_reg(ret.vreg);
                    write!(&mut s, " {vreg}={preg}").unwrap();
                }
                s
            }

            &Inst::Ret {} => "ret".to_string(),

            &MInst::Extend {
                rd,
                rn,
                signed,
                from_bits,
                ..
            } => {
                let rn = format_reg(rn);
                let rd = format_reg(rd.to_reg());
                return if signed == false && from_bits == 8 {
                    format!("andi {rd},{rn}")
                } else {
                    let op = if signed { "srai" } else { "srli" };
                    let shift_bits = (64 - from_bits) as i16;
                    format!("slli {rd},{rn},{shift_bits}; {op} {rd},{rd},{shift_bits}")
                };
            }

            &MInst::Call { ref info } => format!("call {}", info.dest.display(None)),
            &MInst::CallInd { ref info } => {
                let rd = format_reg(info.dest);
                format!("callind {rd}")
            }
            &MInst::ReturnCall { ref info } => {
                let mut s = format!(
                    "return_call {:?} new_stack_arg_size:{}",
                    info.dest, info.new_stack_arg_size
                );
                for ret in &info.uses {
                    let preg = format_reg(ret.preg);
                    let vreg = format_reg(ret.vreg);
                    write!(&mut s, " {vreg}={preg}").unwrap();
                }
                s
            }
            &MInst::ReturnCallInd { ref info } => {
                let callee = format_reg(info.dest);
                let mut s = format!(
                    "return_call_ind {callee} new_stack_arg_size:{}",
                    info.new_stack_arg_size
                );
                for ret in &info.uses {
                    let preg = format_reg(ret.preg);
                    let vreg = format_reg(ret.vreg);
                    write!(&mut s, " {vreg}={preg}").unwrap();
                }
                s
            }
            &MInst::TrapIf {
                rs1,
                rs2,
                cc,
                trap_code,
            } => {
                let rs1 = format_reg(rs1);
                let rs2 = format_reg(rs2);
                format!("trap_if {trap_code}##({rs1} {cc} {rs2})")
            }
            &MInst::Mov { rd, rm, ty } => {
                let rm = format_reg(rm);
                let rd = format_reg(rd.to_reg());

                let op = match ty {
                    F16 => "fmv.h",
                    F32 => "fmv.s",
                    F64 => "fmv.d",
                    ty if ty.is_vector() => "vmv1r.v",
                    _ => "mv",
                };

                format!("{op} {rd},{rm}")
            }
            &MInst::Jal { label } => {
                format!("j {}", label.to_string())
            }

            &MInst::Fence { pred, succ } => {
                format!(
                    "fence {},{}",
                    Inst::fence_req_to_string(pred),
                    Inst::fence_req_to_string(succ),
                )
            }
            &MInst::CondBr {
                taken,
                not_taken,
                kind,
                ..
            } => {
                let rs1 = format_reg(kind.rs1);
                let rs2 = format_reg(kind.rs2);
                if not_taken.is_fallthrouh() && taken.as_label().is_none() {
                    format!("{} {},{},0", kind.op_name(), rs1, rs2)
                } else {
                    let x = format!(
                        "{} {},{},taken({}),not_taken({})",
                        kind.op_name(),
                        rs1,
                        rs2,
                        taken,
                        not_taken
                    );
                    x
                }
            }
            &MInst::LoadExtName {
                rd,
                ref name,
                offset,
            } => {
                let rd = format_reg(rd.to_reg());
                format!("load_sym {},{}{:+}", rd, name.display(None), offset)
            }
            &Inst::ElfTlsGetAddr { rd, ref name } => {
                let rd = format_reg(rd.to_reg());
                format!("elf_tls_get_addr {rd},{}", name.display(None))
            }
            &MInst::LoadAddr { ref rd, ref mem } => {
                let rs = mem.to_string();
                let rd = format_reg(rd.to_reg());
                format!("load_addr {rd},{rs}")
            }
            &MInst::Select {
                ref dst,
                condition,
                ref x,
                ref y,
            } => {
                let c_rs1 = format_reg(condition.rs1);
                let c_rs2 = format_reg(condition.rs2);
                let x = format_regs(x.regs());
                let y = format_regs(y.regs());
                let dst = dst.map(|r| r.to_reg());
                let dst = format_regs(dst.regs());
                format!(
                    "select {},{},{}##condition=({} {} {})",
                    dst,
                    x,
                    y,
                    c_rs1,
                    condition.kind.to_static_str(),
                    c_rs2
                )
            }
            &MInst::Udf { trap_code } => format!("udf##trap_code={trap_code}"),
            &MInst::EBreak {} => String::from("ebreak"),
        }
    }
}

/// Different forms of label references for different instruction formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LabelUse {
    /// 20-bit branch offset (unconditional branches). PC-rel, offset is
    /// imm << 1. Immediate is 20 signed bits. Use in Jal instructions.
    Jal20,

    /// The unconditional jump instructions all use PC-relative
    /// addressing to help support position independent code. The JALR
    /// instruction was defined to enable a two-instruction sequence to
    /// jump anywhere in a 32-bit absolute address range. A LUI
    /// instruction can first load rs1 with the upper 20 bits of a
    /// target address, then JALR can add in the lower bits. Similarly,
    /// AUIPC then JALR can jump anywhere in a 32-bit pc-relative
    /// address range.
    PCRel32,

    /// All branch instructions use the B-type instruction format. The
    /// 12-bit B-immediate encodes signed offsets in multiples of 2, and
    /// is added to the current pc to give the target address. The
    /// conditional branch range is ±4 KiB.
    B12,

    /// Equivalent to the `R_RISCV_PCREL_HI20` relocation, Allows setting
    /// the immediate field of an `auipc` instruction.
    PCRelHi20,

    /// Similar to the `R_RISCV_PCREL_LO12_I` relocation but pointing to
    /// the final address, instead of the `PCREL_HI20` label. Allows setting
    /// the immediate field of I Type instructions such as `addi` or `lw`.
    ///
    /// Since we currently don't support offsets in labels, this relocation has
    /// an implicit offset of 4.
    PCRelLo12I,

    /// 11-bit PC-relative jump offset. Equivalent to the `RVC_JUMP` relocation
    RVCJump,
}

impl MachInstLabelUse for LabelUse {
    /// Alignment for veneer code. Every Riscv32 instruction must be
    /// 4-byte-aligned.
    const ALIGN: CodeOffset = 4;

    /// Maximum PC-relative range (positive), inclusive.
    fn max_pos_range(self) -> CodeOffset {
        match self {
            LabelUse::Jal20 => ((1 << 19) - 1) * 2,
            LabelUse::PCRelLo12I | LabelUse::PCRelHi20 | LabelUse::PCRel32 => {
                Inst::imm_max() as CodeOffset
            }
            LabelUse::B12 => ((1 << 11) - 1) * 2,
            LabelUse::RVCJump => ((1 << 10) - 1) * 2,
        }
    }

    /// Maximum PC-relative range (negative).
    fn max_neg_range(self) -> CodeOffset {
        match self {
            LabelUse::PCRel32 => Inst::imm_min().abs() as CodeOffset,
            _ => self.max_pos_range() + 2,
        }
    }

    /// Size of window into code needed to do the patch.
    fn patch_size(self) -> CodeOffset {
        match self {
            LabelUse::RVCJump => 2,
            LabelUse::Jal20 | LabelUse::B12 | LabelUse::PCRelHi20 | LabelUse::PCRelLo12I => 4,
            LabelUse::PCRel32 => 8,
        }
    }

    /// Perform the patch.
    fn patch(self, buffer: &mut [u8], use_offset: CodeOffset, label_offset: CodeOffset) {
        assert!(use_offset % 2 == 0);
        assert!(label_offset % 2 == 0);
        let offset = (label_offset as i32) - (use_offset as i32);

        // re-check range
        assert!(
            offset >= -(self.max_neg_range() as i32) && offset <= (self.max_pos_range() as i32),
            "{self:?} offset '{offset}' use_offset:'{use_offset}' label_offset:'{label_offset}'  must not exceed max range.",
        );
        self.patch_raw_offset(buffer, offset);
    }

    /// Is a veneer supported for this label reference type?
    fn supports_veneer(self) -> bool {
        match self {
            Self::Jal20 | Self::B12 | Self::RVCJump => true,
            _ => false,
        }
    }

    /// How large is the veneer, if supported?
    fn veneer_size(self) -> CodeOffset {
        match self {
            Self::B12 | Self::Jal20 | Self::RVCJump => 8,
            _ => unreachable!(),
        }
    }

    fn worst_case_veneer_size() -> CodeOffset {
        8
    }

    /// Generate a veneer into the buffer, given that this veneer is at `veneer_offset`, and return
    /// an offset and label-use for the veneer's use of the original label.
    fn generate_veneer(
        self,
        buffer: &mut [u8],
        veneer_offset: CodeOffset,
    ) -> (CodeOffset, LabelUse) {
        let base = writable_spilltmp_reg();
        {
            let x = enc_auipc(base, Imm20::ZERO).to_le_bytes();
            buffer[0] = x[0];
            buffer[1] = x[1];
            buffer[2] = x[2];
            buffer[3] = x[3];
        }
        {
            let x = enc_jalr(writable_zero_reg(), base.to_reg(), Imm12::ZERO).to_le_bytes();
            buffer[4] = x[0];
            buffer[5] = x[1];
            buffer[6] = x[2];
            buffer[7] = x[3];
        }
        (veneer_offset, Self::PCRel32)
    }

    fn from_reloc(reloc: Reloc, addend: Addend) -> Option<LabelUse> {
        match (reloc, addend) {
            (Reloc::RiscvCallPlt, _) => Some(Self::PCRel32),
            _ => None,
        }
    }
}

impl LabelUse {
    #[allow(dead_code)] // in case it's needed in the future
    fn offset_in_range(self, offset: i32) -> bool {
        let min = -(self.max_neg_range() as i32);
        let max = self.max_pos_range() as i32;
        offset >= min && offset <= max
    }

    fn patch_raw_offset(self, buffer: &mut [u8], offset: i32) {
        let insn = match self {
            LabelUse::RVCJump => u16::from_le_bytes(buffer[..2].try_into().unwrap()) as u32,
            _ => u32::from_le_bytes(buffer[..4].try_into().unwrap()),
        };

        match self {
            LabelUse::Jal20 => {
                let offset = offset as u32;
                let v = ((offset >> 12 & 0b1111_1111) << 12)
                    | ((offset >> 11 & 0b1) << 20)
                    | ((offset >> 1 & 0b11_1111_1111) << 21)
                    | ((offset >> 20 & 0b1) << 31);
                buffer[0..4].clone_from_slice(&u32::to_le_bytes(insn | v));
            }
            LabelUse::PCRel32 => {
                let insn2 = u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]);
                Inst::generate_imm(offset as u32)
                    .map(|(imm20, imm12)| {
                        // Encode the OR-ed-in value with zero_reg(). The
                        // register parameter must be in the original
                        // encoded instruction and or'ing in zeroes does not
                        // change it.
                        buffer[0..4].clone_from_slice(&u32::to_le_bytes(
                            insn | enc_auipc(writable_zero_reg(), imm20),
                        ));
                        buffer[4..8].clone_from_slice(&u32::to_le_bytes(
                            insn2 | enc_jalr(writable_zero_reg(), zero_reg(), imm12),
                        ));
                    })
                    // expect make sure we handled.
                    .expect("we have check the range before,this is a compiler error.");
            }

            LabelUse::B12 => {
                let offset = offset as u32;
                let v = ((offset >> 11 & 0b1) << 7)
                    | ((offset >> 1 & 0b1111) << 8)
                    | ((offset >> 5 & 0b11_1111) << 25)
                    | ((offset >> 12 & 0b1) << 31);
                buffer[0..4].clone_from_slice(&u32::to_le_bytes(insn | v));
            }

            LabelUse::PCRelHi20 => {
                // See https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#pc-relative-symbol-addresses
                //
                // We need to add 0x800 to ensure that we land at the next page as soon as it goes out of range for the
                // Lo12 relocation. That relocation is signed and has a maximum range of -2048..2047. So when we get an
                // offset of 2048, we need to land at the next page and subtract instead.
                let offset = offset as u32;
                let hi20 = offset.wrapping_add(0x800) >> 12;
                let insn = (insn & 0xFFF) | (hi20 << 12);
                buffer[0..4].clone_from_slice(&u32::to_le_bytes(insn));
            }

            LabelUse::PCRelLo12I => {
                // `offset` is the offset from the current instruction to the target address.
                //
                // However we are trying to compute the offset to the target address from the previous instruction.
                // The previous instruction should be the one that contains the PCRelHi20 relocation and
                // stores/references the program counter (`auipc` usually).
                //
                // Since we are trying to compute the offset from the previous instruction, we can
                // represent it as offset = target_address - (current_instruction_address - 4)
                // which is equivalent to offset = target_address - current_instruction_address + 4.
                //
                // Thus we need to add 4 to the offset here.
                let lo12 = (offset + 4) as u32 & 0xFFF;
                let insn = (insn & 0xFFFFF) | (lo12 << 20);
                buffer[0..4].clone_from_slice(&u32::to_le_bytes(insn));
            }
            LabelUse::RVCJump => {
                unimplemented!()
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn label_use_max_range() {
        assert!(LabelUse::B12.max_neg_range() == LabelUse::B12.max_pos_range() + 2);
        assert!(LabelUse::Jal20.max_neg_range() == LabelUse::Jal20.max_pos_range() + 2);
        assert!(LabelUse::PCRel32.max_pos_range() == (Inst::imm_max() as CodeOffset));
        assert!(LabelUse::PCRel32.max_neg_range() == (Inst::imm_min().abs() as CodeOffset));
        assert!(LabelUse::B12.max_pos_range() == ((1 << 11) - 1) * 2);
    }
}