import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import asm_identity as ai


def elf(n: int, *, target: int = 2, pool: str = "1", table: tuple[int, int] = (2, 3)) -> str:
    """A function numbered `n` that loads a constant pool emitted before it, branches to
    block `target` and jumps through a table of blocks `table` emitted after it, as
    x86-64 ELF writes them."""
    return f"""\t.section\t.rodata.cst8,"aM",@progbits,8
\t.p2align\t3, 0x0
.LCPI{n}_0:
\t.quad\t{pool}
\t.section\t.text.f,"ax",@progbits
«f»:
\t.cfi_startproc
\tmovsd\t.LCPI{n}_0(%rip), %xmm0
\tjne\t.LBB{n}_{target}
\tjmpq\t*.LJTI{n}_0(,%rax,8)
.LBB{n}_2:
\tretq
.LBB{n}_3:
\tleaq\t.Lanon.0123abcd.{n}(%rip), %rdi
\tretq
.Lfunc_end{n}:
\t.cfi_endproc
\t.section\t.rodata.f,"a",@progbits
\t.p2align\t2, 0x0
.LJTI{n}_0:
\t.long\t.LBB{n}_{table[0]}-.LJTI{n}_0
\t.long\t.LBB{n}_{table[1]}-.LJTI{n}_0
"""


def same(a: str, b: str) -> bool:
    return ai.functions(a, []) == ai.functions(b, [])


def test_a_function_numbered_apart_compares_equal() -> None:
    # Another function index renumbers every label and the anonymous constant.
    assert same(elf(0), elf(7))


def test_a_retargeted_branch_differs() -> None:
    assert not same(elf(0), elf(0, target=3))


def test_swapped_jump_table_entries_differ() -> None:
    assert not same(elf(0), elf(0, table=(3, 2)))


def test_a_changed_constant_pool_differs() -> None:
    assert not same(elf(0), elf(0, pool="2"))


def test_pools_and_tables_follow_the_code() -> None:
    assert ai.functions(elf(0), [])["«f»"] == [
        "movsd\tL2(%rip), %xmm0",
        "jne\tL0",
        "jmpq\t*L3(,%rax,8)",
        "L0:",
        "retq",
        "L1:",
        "leaq\tanon(%rip), %rdi",
        "retq",
        "L2:",
        ".quad\t1",
        "L3:",
        ".long\tL0-L3",
        ".long\tL1-L3",
    ]


def test_mach_o_tables_after_the_function_compare_their_targets() -> None:
    def macho(first: int) -> str:
        return f"""«g»:
\t.cfi_startproc
\tadrp\tx13, LJTI4_0@PAGE
\tadr\tx14, LBB4_2
LBB4_2:
\tret
LBB4_3:
\tret
Lfunc_end4:
\t.cfi_endproc
\t.section\t__TEXT,__const
LJTI4_0:
\t.byte\t(LBB4_{first}-LBB4_2)>>2
"""

    assert same(macho(3), macho(3))
    assert not same(macho(3), macho(2))
