//! RV32 ISA definitions: instruction arguments.

use super::*;
use crate::ir::condcodes::CondCode;

use std::fmt::Result;

/// A macro for defining a newtype of `Reg` that enforces some invariant about
/// the wrapped `Reg` (such as that it is of a particular register class).
macro_rules! newtype_of_reg {
    (
        $newtype_reg:ident,
        $newtype_writable_reg:ident,
        |$check_reg:ident| $check:expr
    ) => {
        /// A newtype wrapper around `Reg`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $newtype_reg(Reg);

        impl PartialEq<Reg> for $newtype_reg {
            fn eq(&self, other: &Reg) -> bool {
                self.0 == *other
            }
        }

        impl From<$newtype_reg> for Reg {
            fn from(r: $newtype_reg) -> Self {
                r.0
            }
        }

        impl $newtype_reg {
            /// Create this newtype from the given register, or return `None` if the register
            /// is not a valid instance of this newtype.
            pub fn new($check_reg: Reg) -> Option<Self> {
                if $check {
                    Some(Self($check_reg))
                } else {
                    None
                }
            }

            /// Get this newtype's underlying `Reg`.
            pub fn to_reg(self) -> Reg {
                self.0
            }
        }

        // Convenience impl so that people working with this newtype can use it
        // "just like" a plain `Reg`.
        //
        // NB: We cannot implement `DerefMut` because that would let people do
        // nasty stuff like `*my_xreg.deref_mut() = some_freg`, breaking the
        // invariants that `XReg` provides.
        impl std::ops::Deref for $newtype_reg {
            type Target = Reg;

            fn deref(&self) -> &Reg {
                &self.0
            }
        }

        /// Writable Reg.
        pub type $newtype_writable_reg = Writable<$newtype_reg>;
    };
}

// Newtypes for registers classes.
newtype_of_reg!(XReg, WritableXReg, |reg| reg.class() == RegClass::Int);
newtype_of_reg!(FReg, WritableFReg, |reg| reg.class() == RegClass::Float);
newtype_of_reg!(VReg, WritableVReg, |reg| reg.class() == RegClass::Vector);

/// An addressing mode specified for a load/store operation.
#[derive(Clone, Debug, Copy)]
pub enum AMode {
    /// Arbitrary offset from a register. Converted to generation of large
    /// offsets with multiple instructions as necessary during code emission.
    RegOffset(Reg, i32),
    /// Offset from the stack pointer.
    SPOffset(i32),

    /// Offset from the frame pointer.
    FPOffset(i32),

    /// Offset into the slot area of the stack, which lies just above the
    /// outgoing argument area that's setup by the function prologue.
    /// At emission time, this is converted to `SPOffset` with a fixup added to
    /// the offset constant. The fixup is a running value that is tracked as
    /// emission iterates through instructions in linear order, and can be
    /// adjusted up and down with [Inst::VirtualSPOffsetAdj].
    ///
    /// The standard ABI is in charge of handling this (by emitting the
    /// adjustment meta-instructions). See the diagram in the documentation
    /// for [crate::isa::aarch64::abi](the ABI module) for more details.
    SlotOffset(i32),

    /// Offset into the argument area.
    IncomingArg(i32),

    /// A reference to a constant which is placed outside of the function's
    /// body, typically at the end.
    Const(VCodeConstant),

    /// A reference to a label.
    Label(MachLabel),
}

impl AMode {
    /// Add the registers referenced by this AMode to `collector`.
    pub(crate) fn get_operands(&mut self, collector: &mut impl OperandVisitor) {
        match self {
            AMode::RegOffset(reg, ..) => collector.reg_use(reg),
            // Registers used in these modes aren't allocatable.
            AMode::SPOffset(..)
            | AMode::FPOffset(..)
            | AMode::SlotOffset(..)
            | AMode::IncomingArg(..)
            | AMode::Const(..)
            | AMode::Label(..) => {}
        }
    }

    pub(crate) fn get_base_register(&self) -> Option<Reg> {
        match self {
            &AMode::RegOffset(reg, ..) => Some(reg),
            &AMode::SPOffset(..) => Some(stack_reg()),
            &AMode::FPOffset(..) => Some(fp_reg()),
            &AMode::SlotOffset(..) => Some(stack_reg()),
            &AMode::IncomingArg(..) => Some(stack_reg()),
            &AMode::Const(..) | AMode::Label(..) => None,
        }
    }

    pub(crate) fn get_offset_with_state(&self, state: &EmitState) -> i32 {
        match self {
            &AMode::SlotOffset(offset) => offset + state.frame_layout().outgoing_args_size as i32,

            // Compute the offset into the incoming argument area relative to SP
            &AMode::IncomingArg(offset) => {
                let frame_layout = state.frame_layout();
                let sp_offset = frame_layout.tail_args_size
                    + frame_layout.setup_area_size
                    + frame_layout.clobber_size
                    + frame_layout.fixed_frame_storage_size
                    + frame_layout.outgoing_args_size;
                sp_offset as i32 - offset
            }

            &AMode::RegOffset(_, offset) => offset,
            &AMode::SPOffset(offset) => offset,
            &AMode::FPOffset(offset) => offset,
            &AMode::Const(_) | &AMode::Label(_) => 0,
        }
    }

    /// Retrieve a MachLabel that corresponds to this addressing mode, if it exists.
    pub(crate) fn get_label_with_sink(&self, sink: &mut MachBuffer<Inst>) -> Option<MachLabel> {
        match self {
            &AMode::Const(addr) => Some(sink.get_label_for_constant(addr)),
            &AMode::Label(label) => Some(label),
            &AMode::RegOffset(..)
            | &AMode::SPOffset(..)
            | &AMode::FPOffset(..)
            | &AMode::IncomingArg(..)
            | &AMode::SlotOffset(..) => None,
        }
    }
}

impl Display for AMode {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        match self {
            &AMode::RegOffset(r, offset, ..) => {
                write!(f, "{}({})", offset, reg_name(r))
            }
            &AMode::SPOffset(offset, ..) => {
                write!(f, "{offset}(sp)")
            }
            &AMode::SlotOffset(offset, ..) => {
                write!(f, "{offset}(slot)")
            }
            &AMode::IncomingArg(offset) => {
                write!(f, "-{offset}(incoming_arg)")
            }
            &AMode::FPOffset(offset, ..) => {
                write!(f, "{offset}(fp)")
            }
            &AMode::Const(addr, ..) => {
                write!(f, "[const({})]", addr.as_u32())
            }
            &AMode::Label(label) => {
                write!(f, "[label{}]", label.as_u32())
            }
        }
    }
}

impl Into<AMode> for StackAMode {
    fn into(self) -> AMode {
        match self {
            StackAMode::IncomingArg(offset, stack_args_size) => {
                AMode::IncomingArg(stack_args_size as i32 - offset as i32)
            }
            StackAMode::OutgoingArg(offset) => AMode::SPOffset(offset as i32),
            StackAMode::Slot(offset) => AMode::SlotOffset(offset as i32),
        }
    }
}

/// risc-v always take two register to compare
#[derive(Clone, Copy, Debug)]
pub struct IntegerCompare {
    pub(crate) kind: IntCC,
    pub(crate) rs1: Reg,
    pub(crate) rs2: Reg,
}

pub(crate) enum BranchFunct3 {
    // ==
    Eq,
    // !=
    Ne,
    // signed <
    Lt,
    // signed >=
    Ge,
    // unsigned <
    Ltu,
    // unsigned >=
    Geu,
}

impl BranchFunct3 {
    pub(crate) fn funct3(self) -> u32 {
        match self {
            BranchFunct3::Eq => 0b000,
            BranchFunct3::Ne => 0b001,
            BranchFunct3::Lt => 0b100,
            BranchFunct3::Ge => 0b101,
            BranchFunct3::Ltu => 0b110,
            BranchFunct3::Geu => 0b111,
        }
    }
}

impl IntegerCompare {
    pub(crate) fn op_code(self) -> u32 {
        0b1100011
    }

    // funct3 and if need inverse the register
    pub(crate) fn funct3(&self) -> (BranchFunct3, bool) {
        match self.kind {
            IntCC::Equal => (BranchFunct3::Eq, false),
            IntCC::NotEqual => (BranchFunct3::Ne, false),
            IntCC::SignedLessThan => (BranchFunct3::Lt, false),
            IntCC::SignedGreaterThanOrEqual => (BranchFunct3::Ge, false),

            IntCC::SignedGreaterThan => (BranchFunct3::Lt, true),
            IntCC::SignedLessThanOrEqual => (BranchFunct3::Ge, true),

            IntCC::UnsignedLessThan => (BranchFunct3::Ltu, false),
            IntCC::UnsignedGreaterThanOrEqual => (BranchFunct3::Geu, false),

            IntCC::UnsignedGreaterThan => (BranchFunct3::Ltu, true),
            IntCC::UnsignedLessThanOrEqual => (BranchFunct3::Geu, true),
        }
    }

    #[inline]
    pub(crate) fn op_name(&self) -> &'static str {
        match self.kind {
            IntCC::Equal => "beq",
            IntCC::NotEqual => "bne",
            IntCC::SignedLessThan => "blt",
            IntCC::SignedGreaterThanOrEqual => "bge",
            IntCC::SignedGreaterThan => "bgt",
            IntCC::SignedLessThanOrEqual => "ble",
            IntCC::UnsignedLessThan => "bltu",
            IntCC::UnsignedGreaterThanOrEqual => "bgeu",
            IntCC::UnsignedGreaterThan => "bgtu",
            IntCC::UnsignedLessThanOrEqual => "bleu",
        }
    }

    pub(crate) fn emit(self) -> u32 {
        let (funct3, reverse) = self.funct3();
        let (rs1, rs2) = if reverse {
            (self.rs2, self.rs1)
        } else {
            (self.rs1, self.rs2)
        };

        self.op_code()
            | funct3.funct3() << 12
            | reg_to_gpr_num(rs1) << 15
            | reg_to_gpr_num(rs2) << 20
    }

    pub(crate) fn inverse(self) -> Self {
        Self {
            kind: self.kind.complement(),
            ..self
        }
    }

    pub(crate) fn regs(&self) -> [Reg; 2] {
        [self.rs1, self.rs2]
    }
}

impl AluOPRRR {
    pub(crate) const fn op_name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Sll => "sll",
            Self::Slt => "slt",
            Self::SltU => "sltu",
            Self::Xor => "xor",
            Self::Srl => "srl",
            Self::Sra => "sra",
            Self::Or => "or",
            Self::And => "and",
        }
    }

    pub fn funct3(self) -> u32 {
        match self {
            AluOPRRR::Add => 0b000,
            AluOPRRR::Sll => 0b001,
            AluOPRRR::Slt => 0b010,

            AluOPRRR::SltU => 0b011,

            AluOPRRR::Xor => 0b100,
            AluOPRRR::Srl => 0b101,
            AluOPRRR::Sra => 0b101,
            AluOPRRR::Or => 0b110,
            AluOPRRR::And => 0b111,
            AluOPRRR::Sub => 0b000,
        }
    }

    pub fn op_code(self) -> u32 {
        match self {
            AluOPRRR::Add
            | AluOPRRR::Sub
            | AluOPRRR::Sll
            | AluOPRRR::Slt
            | AluOPRRR::SltU
            | AluOPRRR::Xor
            | AluOPRRR::Srl
            | AluOPRRR::Sra
            | AluOPRRR::Or
            | AluOPRRR::And => 0b0110011,
        }
    }

    pub const fn funct7(self) -> u32 {
        match self {
            AluOPRRR::Add => 0b0000000,
            AluOPRRR::Sub => 0b0100000,
            AluOPRRR::Sll => 0b0000000,
            AluOPRRR::Slt => 0b0000000,
            AluOPRRR::SltU => 0b0000000,
            AluOPRRR::Xor => 0b0000000,
            AluOPRRR::Srl => 0b0000000,
            AluOPRRR::Sra => 0b0100000,
            AluOPRRR::Or => 0b0000000,
            AluOPRRR::And => 0b0000000,
        }
    }

    pub(crate) fn reverse_rs(self) -> bool {
        false
    }
}

impl AluOPRRI {
    pub(crate) fn option_funct6(self) -> Option<u32> {
        let x: Option<u32> = match self {
            Self::Slli => Some(0b00_0000),
            Self::Srli => Some(0b00_0000),
            Self::Srai => Some(0b01_0000),
            _ => None,
        };
        x
    }

    pub(crate) fn option_funct7(self) -> Option<u32> {
        let x = match self {
            _ => None,
        };
        x
    }

    pub(crate) fn imm12(self, imm12: Imm12) -> u32 {
        let x = imm12.bits();
        if let Some(func) = self.option_funct6() {
            func << 6 | (x & 0b11_1111)
        } else if let Some(func) = self.option_funct7() {
            func << 5 | (x & 0b1_1111)
        } else if let Some(func) = self.option_funct12() {
            func
        } else {
            x
        }
    }

    pub(crate) fn option_funct12(self) -> Option<u32> {
        match self {
            _ => None,
        }
    }

    pub(crate) fn op_name(self) -> &'static str {
        match self {
            Self::Addi => "addi",
            Self::Slti => "slti",
            Self::SltiU => "sltiu",
            Self::Xori => "xori",
            Self::Ori => "ori",
            Self::Andi => "andi",
            Self::Slli => "slli",
            Self::Srli => "srli",
            Self::Srai => "srai",
        }
    }

    pub fn funct3(self) -> u32 {
        match self {
            AluOPRRI::Addi => 0b000,
            AluOPRRI::Slti => 0b010,
            AluOPRRI::SltiU => 0b011,
            AluOPRRI::Xori => 0b100,
            AluOPRRI::Ori => 0b110,
            AluOPRRI::Andi => 0b111,
            AluOPRRI::Slli => 0b001,
            AluOPRRI::Srli => 0b101,
            AluOPRRI::Srai => 0b101,
        }
    }

    pub fn op_code(self) -> u32 {
        match self {
            AluOPRRI::Addi
            | AluOPRRI::Slti
            | AluOPRRI::SltiU
            | AluOPRRI::Xori
            | AluOPRRI::Ori
            | AluOPRRI::Andi
            | AluOPRRI::Slli
            | AluOPRRI::Srli
            | AluOPRRI::Srai => 0b0010011,
        }
    }
}

impl LoadOP {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            Self::Lb => "lb",
            Self::Lh => "lh",
            Self::Lw => "lw",
            Self::Lbu => "lbu",
            Self::Lhu => "lhu",
        }
    }

    pub(crate) fn from_type(ty: Type) -> Self {
        match ty {
            I8 => Self::Lb,
            I16 => Self::Lh,
            I32 => Self::Lw,
            _ => unreachable!(),
        }
    }

    pub(crate) fn size(&self) -> i32 {
        match self {
            Self::Lb | Self::Lbu => 1,
            Self::Lh | Self::Lhu => 2,
            Self::Lw => 4,
        }
    }

    pub(crate) fn op_code(self) -> u32 {
        match self {
            Self::Lb | Self::Lh | Self::Lw | Self::Lbu | Self::Lhu => 0b0000011,
        }
    }
    pub(crate) fn funct3(self) -> u32 {
        match self {
            Self::Lb => 0b000,
            Self::Lh => 0b001,
            Self::Lw => 0b010,
            Self::Lbu => 0b100,
            Self::Lhu => 0b101,
        }
    }
}

impl StoreOP {
    pub(crate) fn op_name(self) -> &'static str {
        match self {
            Self::Sb => "sb",
            Self::Sh => "sh",
            Self::Sw => "sw",
        }
    }
    pub(crate) fn from_type(ty: Type) -> Self {
        match ty {
            I8 => Self::Sb,
            I16 => Self::Sh,
            I32 => Self::Sw,
            _ => unreachable!(),
        }
    }

    pub(crate) fn size(&self) -> i32 {
        match self {
            Self::Sb => 1,
            Self::Sh => 2,
            Self::Sw => 4,
        }
    }

    pub(crate) fn op_code(self) -> u32 {
        match self {
            Self::Sb | Self::Sh | Self::Sw => 0b0100011,
        }
    }
    pub(crate) fn funct3(self) -> u32 {
        match self {
            Self::Sb => 0b000,
            Self::Sh => 0b001,
            Self::Sw => 0b010,
        }
    }
}

impl Inst {
    /// fence request bits.
    pub(crate) const FENCE_REQ_I: u8 = 1 << 3;
    pub(crate) const FENCE_REQ_O: u8 = 1 << 2;
    pub(crate) const FENCE_REQ_R: u8 = 1 << 1;
    pub(crate) const FENCE_REQ_W: u8 = 1 << 0;
    pub(crate) fn fence_req_to_string(x: u8) -> String {
        let mut s = String::default();
        if x & Self::FENCE_REQ_I != 0 {
            s.push_str("i");
        }
        if x & Self::FENCE_REQ_O != 0 {
            s.push_str("o");
        }
        if x & Self::FENCE_REQ_R != 0 {
            s.push_str("r");
        }
        if x & Self::FENCE_REQ_W != 0 {
            s.push_str("w");
        }
        s
    }
}

impl CsrRegOP {
    pub(crate) fn funct3(self) -> u32 {
        match self {
            CsrRegOP::CsrRW => 0b001,
            CsrRegOP::CsrRS => 0b010,
            CsrRegOP::CsrRC => 0b011,
        }
    }

    pub(crate) fn opcode(self) -> u32 {
        0b1110011
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            CsrRegOP::CsrRW => "csrrw",
            CsrRegOP::CsrRS => "csrrs",
            CsrRegOP::CsrRC => "csrrc",
        }
    }
}

impl Display for CsrRegOP {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name())
    }
}

impl CsrImmOP {
    pub(crate) fn funct3(self) -> u32 {
        match self {
            CsrImmOP::CsrRWI => 0b101,
            CsrImmOP::CsrRSI => 0b110,
            CsrImmOP::CsrRCI => 0b111,
        }
    }

    pub(crate) fn opcode(self) -> u32 {
        0b1110011
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            CsrImmOP::CsrRWI => "csrrwi",
            CsrImmOP::CsrRSI => "csrrsi",
            CsrImmOP::CsrRCI => "csrrci",
        }
    }
}

impl Display for CsrImmOP {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name())
    }
}

impl CSR {
    pub(crate) fn bits(self) -> Imm12 {
        Imm12::from_i16(match self {
            CSR::Frm => 0x0002,
        })
    }

    pub(crate) fn name(self) -> &'static str {
        match self {
            CSR::Frm => "frm",
        }
    }
}

impl Display for CSR {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{}", self.name())
    }
}
