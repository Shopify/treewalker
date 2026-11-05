"""Assembly identity preserves branch targets and referenced data."""

import pytest

from scripts.asm_identity import functions


@pytest.mark.parametrize("branches", ["jne .LBB0_3\n jmp .LBB0_3", "jne .LBB0_3\n jmp .LBB0_2"])
def test_branch_targets_are_not_erased(branches):
    text = "«f»:\n jne .LBB0_2\n jmp .LBB0_3\n.LBB0_2:\n ret\n.LBB0_3:\n ret\n.cfi_endproc"
    changed = text.replace("jne .LBB0_2\n jmp .LBB0_3", branches)
    assert functions(text, []) != functions(changed, [])
    assert functions(text, []) == functions(text.replace("LBB0_", "LBB42_"), [])
