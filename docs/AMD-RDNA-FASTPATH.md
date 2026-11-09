# AMD RADV submission batching (runtime experiment)

The Vulkan runtime already coalesces ready, non-conflicting command-buffer batches
into one `vkQueueSubmit`. On ordinary devices it flushes a run at 64 batches.
This fork allows a **bounded opt-in batch limit** only on a Mesa RADV driver
for a listed AMD discrete RDNA3 GPU (Navi 31, 32 or 33). On those GPUs, the
opt-in limit can change host submit overhead and queue latency; no direction or
size of an improvement is assumed.

This is a runtime scheduling setting, **not a shader translation fast path**.
It does not change Metal semantics, reported device capabilities, SPIR-V,
hazard tracking, binary/timeline semaphore conflict checks, wait order,
fences, guest-memory mapping, or per-command-buffer API results.

## Selection and defaults

| Environment variable | Behavior |
|---|---|
| `M2V_AMD_RUN_LIMIT` | Whole number from **1 to 1024**. Effective only for AMD vendor `0x1002`, a device ID in `AMD_RDNA3_DISCRETE`, and `DriverId::MESA_RADV`. Unset or invalid: 64. |
| `M2V_AMD_DIAG=1` | On device creation, reports the Vulkan driver identity, enabled feature count, subgroup size, stand-in flag and effective batch cap. No other value prints this diagnostic. |
| `M2V_SCHED_STATS=1` | Existing scheduler diagnostic. Shows submission count, number of batches, mean submit-call cost, and why each run ended. |

The setting is read once per process. Run separate processes for different
limits. A coalesced run remains bounded to a scheduler pass and is flushed
before the pass releases its lock. A higher cap is not necessarily faster:
it can make semaphore/conflict handling more expensive or increase latency.

The device-list gate does not claim every GFX11 device has identical features.
Vulkan capabilities are still queried from the real physical device and
restricted to the enabled features. The existing optional
`DeviceOptions::rdna3_stand_in` controls *driver facts* separately; the
batch-limit experiment does not enable the stand-in or inherit its claims.
Other Vulkan drivers, including a different ICD on an AMD PCI device, retain
the original 64-batch path.

## Reims-vGPU A/B check

Use an experimental Reims build with **both** `metal2vulkan` and
`metal2vulkan-runtime` from the same tested commit of this branch.
For a Git-pinned Reims tree update both `rev` fields together; for a vendored
Reims tree update `vendor/metal2vulkan` instead. Do not switch the stable VM
binary or release channel to an untested dependency.

Keep the same host kernel, Mesa version, QEMU binary, VM disk, guest state,
display mode, workload and caches for a pair of runs. Launch **one VM at a
time**. The following launch command is a placeholder, not a complete
Reims launch script:

```sh
M2V_AMD_DIAG=1 M2V_SCHED_STATS=1 M2V_AMD_RUN_LIMIT=64 <your-existing-launch-command> 2> baseline.log
M2V_AMD_DIAG=1 M2V_SCHED_STATS=1 M2V_AMD_RUN_LIMIT=128 <your-existing-launch-command> 2> experimental.log
python3 scripts/amd_rdna_compare.py baseline.log experimental.log --json
```

The existing `m2v-sched` log lines are **per-second summaries**, printed only
when there are submissions in that window. The comparator weights the
rounded mean submit-call time by the submitted count and states this
limitation; it does not calculate p95/p99 frame times, guest responsiveness,
GPU execution time or FPS. Collect Reims' existing presentation and
resource-import diagnostics separately. A change should be enabled by default
only after correctness and real RX 7600 A/B evidence support it.

```sh
REIMS_VGPU_LOG=warn,reims_vgpu_vulkan::gpu::timing=debug
```

Before each run, collect the read-only host baseline with
`scripts/amd-rdna-baseline.sh`. Its snapshots (including driver/PCI identity,
local filesystem paths and repository revisions) stay in
`amd-rdna-evidence/`, which is gitignored; review logs before sharing them.

## Correctness boundaries

DMA-BUF file descriptors and offsets identify *kernel-managed backing
ranges*, not physical addresses. Guest virtual addresses, guest physical
addresses, QEMU host virtual pointers, Vulkan memory allocation offsets,
Vulkan buffer device addresses and guest Metal resource identifiers remain
separate address domains. The batch limit changes none of their mappings or
ownership. For larger submission runs, retain the existing validation of
resource lifetimes, aliasing, synchronization and semaphore ordering.

Run non-GPU gates:

```sh
rustfmt --edition 2021 --check crates/metal2vulkan-runtime/src/amd_rdna.rs
cargo check -p metal2vulkan-runtime
cargo test -p metal2vulkan-runtime --lib amd_rdna
bash -n scripts/amd-rdna-baseline.sh
python3 -m unittest discover -s scripts -p 'test_amd_rdna_*.py' -v
```

The CI workflow `.github/workflows/amd-rdna-fastpath.yml` checks the
build, gating tests, shell script and comparator. A passing CI run
establishes only the portable code checks, not performance or correctness
on a physical RX 7600.
