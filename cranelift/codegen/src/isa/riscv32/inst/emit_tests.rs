#[allow(unused)]
use crate::ir::LibCall;
use crate::isa::riscv32::inst::*;
use std::borrow::Cow;

fn fa7() -> Reg {
    f_reg(17)
}

#[test]
fn test_riscv32_binemit() {
    struct TestUnit {
        inst: Inst,
        assembly: &'static str,
        code: TestEncoding,
    }

    struct TestEncoding(Cow<'static, str>);

    impl From<&'static str> for TestEncoding {
        fn from(value: &'static str) -> Self {
            Self(value.into())
        }
    }

    impl From<u32> for TestEncoding {
        fn from(value: u32) -> Self {
            let value = value.swap_bytes();
            let value = format!("{value:08X}");
            Self(value.into())
        }
    }

    impl TestUnit {
        fn new(inst: Inst, assembly: &'static str, code: impl Into<TestEncoding>) -> Self {
            let code = code.into();
            Self {
                inst,
                assembly,
                code,
            }
        }
    }

    let mut insns = Vec::<TestUnit>::with_capacity(500);

    insns.push(TestUnit::new(Inst::Ret {}, "ret", 0x00008067));

    

    

    //
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Add,
            rd: writable_fp_reg(),
            rs1: fp_reg(),
            rs2: zero_reg(),
        },
        "add fp,fp,zero",
        0x40433,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Addi,
            rd: writable_fp_reg(),
            rs: stack_reg(),
            imm12: Imm12::maybe_from_u64(100).unwrap(),
        },
        "addi fp,sp,100",
        0x6410413,
    ));
    insns.push(TestUnit::new(
        Inst::Lui {
            rd: writable_zero_reg(),
            imm: Imm20::from_i32(120),
        },
        "lui zero,120",
        0x78037,
    ));
    insns.push(TestUnit::new(
        Inst::Auipc {
            rd: writable_zero_reg(),
            imm: Imm20::from_i32(120),
        },
        "auipc zero,120",
        0x78017,
    ));

    insns.push(TestUnit::new(
        Inst::Jalr {
            rd: writable_a0(),
            base: a0(),
            offset: Imm12::from_i16(100),
        },
        "jalr a0,100(a0)",
        0x6450567,
    ));

    insns.push(TestUnit::new(
        Inst::Load {
            rd: writable_a0(),
            op: LoadOP::Lb,
            flags: MemFlags::new(),
            from: AMode::RegOffset(a1(), 100),
        },
        "lb a0,100(a1)",
        0x6458503,
    ));
    insns.push(TestUnit::new(
        Inst::Load {
            rd: writable_a0(),
            op: LoadOP::Lh,
            flags: MemFlags::new(),
            from: AMode::RegOffset(a1(), 100),
        },
        "lh a0,100(a1)",
        0x6459503,
    ));

    insns.push(TestUnit::new(
        Inst::Load {
            rd: writable_a0(),
            op: LoadOP::Lw,
            flags: MemFlags::new(),
            from: AMode::RegOffset(a1(), 100),
        },
        "lw a0,100(a1)",
        0x645a503,
    ));

    
    insns.push(TestUnit::new(
        Inst::Store {
            to: AMode::SPOffset(100),
            op: StoreOP::Sb,
            flags: MemFlags::new(),
            src: a0(),
        },
        "sb a0,100(sp)",
        0x6a10223,
    ));
    insns.push(TestUnit::new(
        Inst::Store {
            to: AMode::SPOffset(100),
            op: StoreOP::Sh,
            flags: MemFlags::new(),
            src: a0(),
        },
        "sh a0,100(sp)",
        0x6a11223,
    ));
    insns.push(TestUnit::new(
        Inst::Store {
            to: AMode::SPOffset(100),
            op: StoreOP::Sw,
            flags: MemFlags::new(),
            src: a0(),
        },
        "sw a0,100(sp)",
        0x6a12223,
    ));
    
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Addi,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(100),
        },
        "addi a0,a0,100",
        0x6450513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Slti,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(100),
        },
        "slti a0,a0,100",
        0x6452513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::SltiU,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(100),
        },
        "sltiu a0,a0,100",
        0x6453513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Xori,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(100),
        },
        "xori a0,a0,100",
        0x6454513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Andi,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(100),
        },
        "andi a0,a0,100",
        0x6457513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Slli,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(5),
        },
        "slli a0,a0,5",
        0x551513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Srli,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(5),
        },
        "srli a0,a0,5",
        0x555513,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRImm12 {
            alu_op: AluOPRRI::Srai,
            rd: writable_a0(),
            rs: a0(),
            imm12: Imm12::from_i16(5),
        },
        "srai a0,a0,5",
        0x40555513,
    ));
    
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Add,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "add a0,a0,a1",
        0xb50533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Sub,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "sub a0,a0,a1",
        0x40b50533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Sll,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "sll a0,a0,a1",
        0xb51533,
    ));

    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Slt,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "slt a0,a0,a1",
        0xb52533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::SltU,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "sltu a0,a0,a1",
        0xb53533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Xor,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "xor a0,a0,a1",
        0xb54533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Srl,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "srl a0,a0,a1",
        0xb55533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Sra,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "sra a0,a0,a1",
        0x40b55533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::Or,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "or a0,a0,a1",
        0xb56533,
    ));
    insns.push(TestUnit::new(
        Inst::AluRRR {
            alu_op: AluOPRRR::And,
            rd: writable_a0(),
            rs1: a0(),
            rs2: a1(),
        },
        "and a0,a0,a1",
        0xb57533,
    ));
    
    let (flags, isa_flags) = make_test_flags();
    let emit_info = EmitInfo::new(flags, isa_flags);

    for unit in insns.iter() {
        println!("Riscv32: {:?}, {}", unit.inst, unit.assembly);
        // Check the printed text is as expected.
        let actual_printing = unit.inst.print_with_state(&mut EmitState::default());
        assert_eq!(unit.assembly, actual_printing);
        let mut buffer = MachBuffer::new();
        unit.inst
            .emit(&mut buffer, &emit_info, &mut Default::default());
        let buffer = buffer.finish(&Default::default(), &mut Default::default());
        let actual_encoding = buffer.stringify_code_bytes();

        assert_eq!(actual_encoding, unit.code.0);
    }
}

fn make_test_flags() -> (settings::Flags, super::super::riscv_settings::Flags) {
    let b = settings::builder();
    let flags = settings::Flags::new(b.clone());
    let b2 = super::super::riscv_settings::builder();
    let isa_flags = super::super::riscv_settings::Flags::new(&flags, &b2);
    (flags, isa_flags)
}

