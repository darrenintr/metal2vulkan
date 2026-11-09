//! Opt-in RADV/RDNA3 submission experiments. No Metal API or shader answers change here.
//!
//! AMD's Vulkan driver is not identified by the PCI vendor alone: an AMD GPU can run
//! another ICD. An experiment is enabled only for the explicitly listed discrete GFX11
//! devices on Mesa RADV, never for a lookalike integrated GPU or another Vulkan driver.
//! No guessed capabilities or stolen driver facts are installed by this module.

use metal2vulkan::device::{DeviceDescription, AMD_RDNA3_DISCRETE};
use metal2vulkan::vk::DriverId;
use std::sync::OnceLock;

/// Upstream's proven generic batching limit, unchanged unless explicitly overridden.
pub(crate) const DEFAULT_RUN_LIMIT: usize = 64;

/// Largest permitted *experimental* coalesced batch count. A bounded run must be
/// submitted before the scheduler releases its lock; it never spans a pass.
const MAX_RUN_LIMIT: usize = 1024;

/// True only for a discrete RDNA3 graphics function served by RADV.
///
/// PCI IDs by themselves are not feature assertions. Every shader and resource
/// still follows the queried, enabled Vulkan capabilities and normal validation.
pub(crate) fn is_radv_rdna3(description: &DeviceDescription) -> bool {
    description.vendor_id == 0x1002
        && AMD_RDNA3_DISCRETE.contains(&description.device_id)
        && description.driver_id == Some(DriverId::MESA_RADV)
}

/// Parse one bounded experiment value; an invalid or empty setting is ignored.
fn parse_run_limit(value: Option<&str>) -> Option<usize> {
    value
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| (1..=MAX_RUN_LIMIT).contains(&n))
}

/// A pure decision, tested on non-AMD and non-RADV identities as well as AMD.
fn selected_run_limit(description: &DeviceDescription, requested: Option<usize>) -> usize {
    if !is_radv_rdna3(description) {
        return DEFAULT_RUN_LIMIT;
    }
    requested
        .filter(|&n| (1..=MAX_RUN_LIMIT).contains(&n))
        .unwrap_or(DEFAULT_RUN_LIMIT)
}

/// The operator's choice, read once for the lifetime of a process.
///
/// A value not in 1..=1024, or no value at all, preserves the generic limit.
fn configured_run_limit() -> Option<usize> {
    static REQUESTED: OnceLock<Option<usize>> = OnceLock::new();
    *REQUESTED.get_or_init(|| parse_run_limit(std::env::var("M2V_AMD_RUN_LIMIT").ok().as_deref()))
}

/// The maximum number of ready, non-conflicting batches in one vkQueueSubmit.
///
/// The normal coalescer's ordering, dependency, fence and binary semaphore
/// checks remain unchanged. Only the point at which it flushes changes.
pub(crate) fn run_limit(description: &DeviceDescription) -> usize {
    if !is_radv_rdna3(description) {
        return DEFAULT_RUN_LIMIT;
    }
    selected_run_limit(description, configured_run_limit())
}

/// One opt-in diagnostic line per runtime device. It prints the actual Vulkan
/// identity and a selected limit, not an unsupported performance claim.
pub(crate) fn diagnostic(description: &DeviceDescription) -> Option<String> {
    if std::env::var("M2V_AMD_DIAG").ok().as_deref() != Some("1") {
        return None;
    }
    let applicable = is_radv_rdna3(description);
    let requested = if applicable {
        configured_run_limit()
    } else {
        None
    };
    let limit = selected_run_limit(description, requested);
    let mode = if applicable && requested.is_some() {
        "opt-in RADV/RDNA3 batch-limit experiment"
    } else {
        "generic upstream scheduler"
    };
    Some(format!(
        "metal2vulkan: AMD scheduler profile: {mode}; vendor={:#06x} device={:#06x} \
         driver={:?} driverVersion={:#x}; subgroupSize={}; enabledFeatures={} \
         factStandIn={}; batchesPerSubmitLimit={limit}; M2V_SCHED_STATS=1 logs the measured submits",
        description.vendor_id,
        description.device_id,
        description.driver_id,
        description.driver_version,
        description.subgroup.size,
        description.features.len(),
        description.facts_stand_in.is_some()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal2vulkan::device::ApiVersion;

    fn radv_rx7600() -> DeviceDescription {
        let mut description = DeviceDescription::new(ApiVersion::V1_2);
        description.vendor_id = 0x1002;
        description.device_id = 0x7480;
        description.driver_id = Some(DriverId::MESA_RADV);
        description
    }

    #[test]
    fn applies_only_to_discrete_rdna3_on_radv() {
        let mut description = radv_rx7600();
        assert!(is_radv_rdna3(&description));
        assert_eq!(selected_run_limit(&description, Some(128)), 128);

        description.device_id = 0x744c; // Navi 31 (RX 7900 XTX).
        assert!(is_radv_rdna3(&description));
        description.device_id = 0x15bf; // GFX11 integrated 780M, not discrete.
        assert!(!is_radv_rdna3(&description));
        assert_eq!(selected_run_limit(&description, Some(128)), 64);

        description.device_id = 0x7480;
        description.driver_id = None;
        assert!(!is_radv_rdna3(&description));
        description.driver_id = Some(DriverId::MESA_LLVMPIPE);
        assert!(!is_radv_rdna3(&description));
        description.driver_id = Some(DriverId::MESA_RADV);
        description.vendor_id = 0x10de;
        assert!(!is_radv_rdna3(&description));
    }

    #[test]
    fn rejects_missing_out_of_range_and_malformed_values() {
        for input in [None, Some(""), Some("0"), Some("1025"), Some("-1"), Some("abc")] {
            assert_eq!(parse_run_limit(input), None);
        }
        assert_eq!(parse_run_limit(Some(" 1 ")), Some(1));
        assert_eq!(parse_run_limit(Some("64")), Some(64));
        assert_eq!(parse_run_limit(Some("1024")), Some(1024));
        let description = radv_rx7600();
        assert_eq!(selected_run_limit(&description, None), 64);
        assert_eq!(selected_run_limit(&description, Some(0)), 64);
        assert_eq!(selected_run_limit(&description, Some(1025)), 64);
        assert_eq!(selected_run_limit(&description, Some(1)), 1);
        assert_eq!(selected_run_limit(&description, Some(1024)), 1024);
    }
}
