"""Patch the vendored LTX-2 source to break a circular import.

Run inside the Modal image build (not against the local submodule):

    python patch_ltx_circular_import.py /opt/ltx2

The bug
-------
`ltx_core/loader/__init__.py` eagerly does `from ...fuse_loras import
apply_loras`. `fuse_loras.py` in turn imports, at module scope:

    from ltx_core.quantization.fp8_cast import fused_add_round_launch
    from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor

`fp8_cast.py` imports `ltx_core.loader.kernels`, which forces
`loader/__init__.py` to run — looping back into a half-initialized
`fp8_cast`. So `import ltx_core.quantization` (required to construct a
`QuantizationPolicy` for FP8) fails:

    ImportError: cannot import name 'fused_add_round_launch' from
    partially initialized module 'ltx_core.quantization.fp8_cast'

The fix
-------
Both imported names are only referenced *inside functions* in
`fuse_loras.py` (`fused_add_round_launch` in `fuse_cast_fp8_weight`,
`quantize_weight_to_fp8_per_tensor` one line below `new_fp8_weight = ...`).
Moving the imports into those function bodies makes them lazy: by the time
the functions run, `fp8_cast` is fully initialized, so the cycle never
triggers. Behavior is unchanged.

This is idempotent — re-running on an already-patched file is a no-op.
"""

import sys
from pathlib import Path

MODULE_IMPORT_FP8_CAST = (
    "from ltx_core.quantization.fp8_cast import fused_add_round_launch\n"
)
MODULE_IMPORT_SCALED_MM = (
    "from ltx_core.quantization.fp8_scaled_mm import "
    "quantize_weight_to_fp8_per_tensor\n"
)

# Anchor lines inside the functions that use each name. The lazy import is
# inserted immediately before its anchor, at the same indentation.
ANCHOR_FP8_CAST = (
    '    if str(weight_fp8.device).startswith("cuda") and TRITON_AVAILABLE:'
)
ANCHOR_SCALED_MM = (
    "    new_fp8_weight, new_weight_scale = "
    "quantize_weight_to_fp8_per_tensor(new_weight)"
)


def patch(ltx_src_dir: str) -> None:
    path = (
        Path(ltx_src_dir)
        / "packages/ltx-core/src/ltx_core/loader/fuse_loras.py"
    )
    src = path.read_text()
    lines = src.splitlines(keepends=True)

    # The two cyclic imports as exact, column-0 lines. Working line-by-line
    # (not substring `in`) is essential: the indented lazy form is a
    # substring of nothing-vs-the-module form only differs by leading
    # whitespace, so substring checks are ambiguous.
    module_lines = {MODULE_IMPORT_FP8_CAST, MODULE_IMPORT_SCALED_MM}
    has_module_level = any(line in module_lines for line in lines)

    if not has_module_level:
        print("[patch] fuse_loras.py already patched (no module-level cyclic import) — skipping")
        return

    # Drop the module-level imports (exact column-0 line match only).
    src = "".join(line for line in lines if line not in module_lines)

    # Re-insert each lazily, immediately before the line that uses it,
    # at function-body indentation. Skip if a lazy copy is already present
    # (defensive — shouldn't happen given the has_module_level guard).
    lazy_fp8_cast = "    " + MODULE_IMPORT_FP8_CAST
    if lazy_fp8_cast not in src:
        if ANCHOR_FP8_CAST not in src:
            raise RuntimeError(f"anchor not found for fp8_cast import in {path}")
        src = src.replace(ANCHOR_FP8_CAST, lazy_fp8_cast + ANCHOR_FP8_CAST, 1)

    lazy_scaled_mm = "    " + MODULE_IMPORT_SCALED_MM
    if lazy_scaled_mm not in src:
        if ANCHOR_SCALED_MM not in src:
            raise RuntimeError(f"anchor not found for scaled_mm import in {path}")
        src = src.replace(ANCHOR_SCALED_MM, lazy_scaled_mm + ANCHOR_SCALED_MM, 1)

    # Sanity check: no module-level (column-0) cyclic import survives.
    if any(line in module_lines for line in src.splitlines(keepends=True)):
        raise RuntimeError(f"patch failed: cyclic import still at module scope in {path}")

    path.write_text(src)
    print(f"[patch] fuse_loras.py circular import fixed: {path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_ltx_circular_import.py <ltx_src_dir>")
    patch(sys.argv[1])
