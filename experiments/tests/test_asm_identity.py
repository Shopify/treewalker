"""Assembly identity preserves branch targets and referenced data."""

import pytest
from asm_identity import functions


@pytest.mark.parametrize("branches", ["jne .LBB0_3\n jmp .LBB0_3", "jne .LBB0_3\n jmp .LBB0_2"])
def test_branch_targets_are_not_erased(branches):
    text = "«f»:\n jne .LBB0_2\n jmp .LBB0_3\n.LBB0_2:\n ret\n.LBB0_3:\n ret\n.cfi_endproc"
    changed = text.replace("jne .LBB0_2\n jmp .LBB0_3", branches)
    assert functions(text, []) != functions(changed, [])
    assert functions(text, []) == functions(text.replace("LBB0_", "LBB42_"), [])


@pytest.mark.parametrize("label", [".LCPI0_0", "lCPI0_0", "LCPI0_0", ".LJTI0_0", "LJTI0_0"])
@pytest.mark.parametrize("placement", ["before", "after", "section"])
def test_referenced_data_is_compared(label, placement):
    body = f"«f»:\n load {label}\n.LBB0_2:\n ret\n.LBB0_3:\n ret\n.cfi_endproc\n"
    pool = f"{label}:\n .quad 1\n .long .LBB0_2-{label}\n"
    text = pool + body if placement == "before" else body + pool
    if placement == "section":
        text = body + ".section .rodata\n" + pool
    assert functions(text, []) != functions(text.replace(".quad 1", ".quad 2"), [])
    assert functions(text, []) != functions(text.replace(".long .LBB0_2", ".long .LBB0_3"), [])
    assert functions(text, []) == functions(text.replace("0_", "42_"), [])


def test_pool_references_keep_their_targets():
    text = "«f»:\n load .LCPI0_0\n ret\n.cfi_endproc\n.LCPI0_0:\n .quad 1\n.LCPI0_1:\n .quad 2\n"
    assert functions(text, []) != functions(text.replace("load .LCPI0_0", "load .LCPI0_1"), [])
