#!/usr/bin/env bash
# Capture a reproducible Linux/RADV baseline for the AMD RDNA3 metal2vulkan experiment.
# Read-only: no sysctl, module, GPU driver, VM or Vulkan setting is modified.
# Usage: bash scripts/amd-rdna-baseline.sh [--out DIRECTORY] [--reims-dir DIRECTORY]
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/amd-rdna-baseline.sh [--out DIRECTORY] [--reims-dir DIRECTORY]
Capture OS, Mesa/RADV, Vulkan, AMD PCI ID, kernel and repository revisions.

No root privileges, GPU tuning or VM launch is required. Results stay on your computer.
The output may include usernames, system paths and hardware identifiers; review before sharing.
EOF
}

out=''
reims_dir=''
while (($#)); do
  case "$1" in
    --out|--reims-dir)
      if (($# < 2)); then
        echo "Missing value for $1" >&2
        exit 2
      fi
      if [[ $1 == --out ]]; then out=$2; else reims_dir=$2; fi
      shift 2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z $out ]]; then
  out="$repo_dir/amd-rdna-evidence/$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$out"
out="$(cd "$out" && pwd)"

run_capture() {
  local name=$1
  shift
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Unavailable: %s\n' "$1" > "$out/$name.txt"
    return 0
  fi
  "$@" > "$out/$name.txt" 2>&1 || {
    local status=$?
    printf '\nCommand exited %s\n' "$status" >> "$out/$name.txt"
  }
}

{
  printf 'Captured UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Purpose: RX 7600 / RADV / metal2vulkan baseline (read-only)\n'
  printf 'Metal2Vulkan working tree: %s\n' "$repo_dir"
  printf 'Reims working tree: %s\n' "${reims_dir:-not supplied}"
  printf 'Baseline comparison: frozen Reims and metal2vulkan revisions, same VM settings\n'
  printf 'No correctness or speedup claim is inferred from these system details.\n'
} > "$out/README.txt"

run_capture kernel uname -a
run_capture vulkan-summary vulkaninfo --summary
run_capture pci-amd lspci -nnk -d 1002:
run_capture mesa-packages dpkg-query -W -f='${binary:Package} ${Version}\n' mesa-vulkan-drivers libvulkan1 vulkan-tools
run_capture metal2vulkan-rev git -C "$repo_dir" log -1 --format='%H%n%s%n%ci'
run_capture metal2vulkan-status git -C "$repo_dir" status --short
if [[ -n $reims_dir ]]; then
  if [[ -d $reims_dir/.git || -f $reims_dir/.git ]]; then
    run_capture reims-rev git -C "$reims_dir" log -1 --format='%H%n%s%n%ci'
    run_capture reims-status git -C "$reims_dir" status --short
  else
    echo "Reims directory is not a Git checkout: $reims_dir" > "$out/reims-rev.txt"
  fi
fi

# Sysfs is intentionally read without root. These are host/PCI identities,
# not guest addresses, host physical addresses or Vulkan device addresses.
{
  shopt -s nullglob
  for card in /sys/class/drm/card[0-9]*; do
    [[ -d "$card/device" ]] || continue
    [[ -r "$card/device/vendor" ]] || continue
    [[ $(< "$card/device/vendor") == 0x1002 ]] || continue
    printf '%s\n' "$card"
    for key in vendor device revision subsystem_vendor subsystem_device; do
      if [[ -r "$card/device/$key" ]]; then
        printf '  %s: %s\n' "$key" "$(< "$card/device/$key")"
      fi
    done
  done
} > "$out/amd-drm-devices.txt"

# Include kernel amdgpu parameters only, when visible; no root access or changes.
{
  shopt -s nullglob
  for param in /sys/module/amdgpu/parameters/*; do
    [[ -f "$param" && -r "$param" ]] || continue
    value="$(cat "$param" 2>/dev/null || true)"
    printf '%s=%s\n' "${param##*/}" "$value"
  done
} > "$out/amdgpu-module-parameters.txt"

cat <<EOF
Wrote read-only baseline snapshot to:
  $out

Next: capture Reims guest timings using:
  REIMS_VGPU_LOG=warn,reims_vgpu_vulkan::gpu::timing=debug

Record cold vs warm shader/pipeline caches, p95/p99 frame times,
submission latency and correctness before enabling an AMD-specific fast path.
EOF
