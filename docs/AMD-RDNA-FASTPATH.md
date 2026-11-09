# AMD RDNA3 fast-path experiment

Status: **experimental plan only**. Creating this branch does **not** enable new runtime fast paths or establish a performance gain.

## Baseline and integration

- Repository: `darrenintr/metal2vulkan`
- Branch: `feature/amd-rdna3-fastpath`
- Starting commit: `6d6f5d92326361562ac5d482eb017f41d80c82d9`
- First target: RX 7600 (Navi 33 / GFX11), Linux Mesa RADV, Reims-vGPU with an x86-64 macOS guest.
- Comparand: the **same** Reims guest, QEMU, Mesa, kernel, display, and VM configuration, using the pinned baseline above.
- Reims `master` currently pins both `metal2vulkan` and `metal2vulkan-runtime` to the fork at the starting commit. Reims `testing` uses a vendored local metal2vulkan path; it must be tested separately.
- No change to Reims or its production dependencies is part of this document.

### Existing work to preserve (do not repeat)

- RDNA3 discrete GPU IDs and an explicitly opt-in GFX11 stand-in for unmeasured devices in `crates/metal2vulkan/src/device/gpus.rs`.
- Shared DMA-BUF imports and offsets within imported DMA-BUF chunks.
- Shader stage key caching, discrete-GPU zero fill and adjacent tiled-color staging reuse already selected into the pinned fork.
- Upstream Vulkan runtime disk/memory shader caching, driver pipeline caches, pipeline-record warmup, existing scheduler coalescing, and device-policy fast paths.
- Reims' GPU-timing and resource-import diagnostics in `crates/reims-vgpu-vulkan/src/gpu/timing.rs` and `resources.rs`.
- The experimental `M2V_DEFER_COUNT`, `M2V_DEFER_LIMIT_US` and `M2V_RUN_LIMIT` scheduler-tuning patch in the Reims repository. Check whether applied before measuring: presence of a patch file does not imply a given binary contains it.

## Operating principles

1. **Measure before rewriting.** Establish whether a frame is blocked by AIR->SPIR-V translation, Vulkan pipeline creation, descriptor/bind work, queue submission/completion, GPU execution, host-memory import/copy, or presentation.
2. **Preserve Metal semantics.** A successful SPIR-V validation is not an execution-correctness check. Use the project's authored conformance tests and a Reims visual/functional regression suite.
3. **Preserve 32-lane Metal SIMD semantics.** Do not substitute the driver's subgroup size unless a separately proven lowering preserves them. There is no generic wave64 shortcut.
4. **No unconditional AMD workarounds.** Detect the selected Vulkan physical device's vendor/driver and check feature/format support at runtime. The PCI device ID is not a complete capability description. Unsupported or unmeasured cases retain the generic path.
5. **Keep rollback obvious.** Each fast path must have an explicit opt-out, a test, diagnostic counters and a documented fallback. Never hide a correctness difference.
6. **Keep provenance clear.** No captured guest metallibs, generated consumer SPIR-V or copyrighted Apple artifacts in git.

## Address and memory domains (never conflate)

- **Guest CPU virtual address:** a macOS guest pointer, interpreted by the guest OS.
- **Guest physical address (GPA):** guest-visible RAM address. QEMU maps this into host virtual memory; a GPA is not a host physical or GPU address.
- **Host CPU virtual address (HVA):** pointer in QEMU/Reims' process for mapped guest RAM.
- **Host physical pages / DMA-BUF:** kernel-managed backing. A DMA-BUF file descriptor plus offset and length identifies a shareable range; it is *not* a physical address.
- **Vulkan buffer/image memory offset:** placement inside a `VkDeviceMemory` allocation. This is not a guest address.
- **Vulkan buffer device address:** returned by `vkGetBufferDeviceAddress`; valid in that selected Vulkan device's GPU virtual-address space and subject to resource lifetime/residency.
- **Emulated Metal GPU address/resource ID:** an address or identifier in the runtime's Metal object space. It must be resolved to the backing Vulkan resource via the runtime's explicit mapping.

Correctness invariants: check alignment, boundaries, lifetimes, aliases, guest writes, GPU reads/writes, cache coherence, and the order of fences/semaphores before reusing imported memory. Do not assume host-pointer import works on shared memfd mappings: RADV's userptr restrictions are already handled through DMA-BUF-backed imports where possible.

## Work packages (one independent experiment per commit)

### P0. Baseline evidence

- Capture Mesa/RADV, amdgpu kernel, PCI ID, `vulkaninfo`, Reims commit, metal2vulkan commit, output mode, VM RAM and video mode.
- Use existing Reims diagnostic target: `REIMS_VGPU_LOG=warn,reims_vgpu_vulkan::gpu::timing=debug`.
- Capture pipeline create/first use, guest command-buffer commit, submit-to-completion, memory imports, CPU upload calls, host staging copies, and frame present cadence.
- Compare *cold* and *warm* shader cache without conflating translation and native Vulkan pipeline compilation.
- Record median/p95/p99 frame time, actual present rate, stutters > 33.3 ms, cache hit rate, and visual/correctness faults. If counters are missing, add instrumentation before optimizing.

### P1. RADV/Navi 33 capability profile

- Query the actual selected Vulkan physical device, driver ID, PCI ID, memory heaps/types, subgroup capabilities, `external_memory_dma_buf`, external-memory-host capabilities, timeline semaphores and relevant formats.
- Compare this capture against the existing GFX11 stand-in. Replace unverified assumptions with measured device capabilities; no fake hardware features.
- Add a test that leaves non-RADV devices on the unchanged generic path.

### P2. DMA-BUF and discrete-VRAM traffic

- Measure frequency and latency of import, re-import, copy, map/unmap, staging upload and invalidation.
- Experiment with import reuse and correctly aligned offset windows before trying larger import chunks. Include memory-pressure cases and resource release.
- Compare zero-copy versus staging only for workloads where both routes are correct and supported.
- Preserve explicit CPU↔GPU ownership/synchronization and avoid unsynchronized reuse of aliases.

### P3. Pipeline preparation and shader throughput

- Verify whether stage-key memoization, disk translation cache, `VkPipelineCache` and `Device::warm` are already used by Reims, including warm restart.
- Record stage translation time separately from first `VkPipeline` creation, driver compile time and first dispatch/draw.
- Consider asynchronous *preparation* only where dependency and first-use ordering remain correct. Never drop a draw silently to gain apparent FPS.
- Inspect generated SPIR-V and RADV/ACO shader statistics only after isolating a costly class of shaders.

### P4. Queue submission and frame pacing

- Compare submit count, submission batch sizes, queue idle gaps, timeline wait latency and display present intervals.
- Upstream already has a measured RADV grouped-submission path: verify it is active before proposing a replacement.
- Test scheduler defer/coalesce parameters one at a time; include low-latency input and deadlock / wait regression cases.
- Prefer reducing excessive submissions/host waits to introducing arbitrary buffering or delayed completion.

### P5. Optional shader-specialized fast paths

- Only after P0–P4, consider safe rewrites for AMD-native SPIR-V features, subgroup operations, half arithmetic and descriptor access patterns.
- Preserve Metal's 32-lane simdgroup meaning, float controls, memory ordering and observable outcomes.
- A new precision tradeoff requires an explicit policy switch, `PrecisionNote`, updated `docs/FAST-PATHS.md`, conformance cases and A/B numbers.

## A/B and promotion gates

For each experiment use three states: baseline, feature enabled, feature disabled. Keep VM and host software identical. Collect repeated runs, not a single FPS snapshot.

- **Primary:** better p95/p99 frame times and fewer long stalls in normal macOS UI movement.
- **Secondary:** higher sustained present rate, lower GPU submission CPU cost, reduced guest-memory copy/import cost and faster cold/warm pipeline startup.
- **Mandatory correctness:** no regression in synthetic Metal↔Vulkan checks, macOS guest rendering, window/content updates, memory aliasing, GPU lifetime and synchronization.
- Report exact test hardware, Mesa, kernel, commits, sample size, results and confidence; do not claim a percentage speedup until measured.
- The default path stays unchanged until the optimization has passed both correctness and real RX 7600 tests.

## Integration notes

On an experimental Reims worktree (not stable/master), pin **both** crates to the same new commit SHA:

```toml
metal2vulkan-runtime = { git = "https://github.com/darrenintr/metal2vulkan", rev = "<tested-amd-fastpath-sha>" }
metal2vulkan = { git = "https://github.com/darrenintr/metal2vulkan", rev = "<tested-amd-fastpath-sha>" }
```

For Reims `testing` with `vendor/metal2vulkan`, change the vendor checkout/lockfile instead of editing in conflicting git dependencies. Save a named release/build for A/B and keep the last known-good build.

## Suggested first implementation

**Instrument + capability audit first.** Build a small RADV-only diagnostic report from the selected `VkPhysicalDevice` and existing runtime counters. This has a bounded scope, should not alter rendering, and creates the evidence required to prioritize memory, scheduler or shader work. Then take the highest-cost measured area as the first isolated fast-path patch.
