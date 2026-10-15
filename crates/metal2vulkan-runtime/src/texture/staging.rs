//! Texture copies' host-visible buffers. A completed synchronous copy may return its buffer to
//! the device; pending work and deferred copies never put a buffer in this pool. Idle buffers
//! hold at most 64 MiB in 16 objects, with no allocation over 16 MiB kept. Each lease is exclusive.

use super::super::device::lock;
use super::super::error::{vulkan, Error, MemoryKind};
use super::super::exit_guard;
use super::super::memory::{Memory, MemoryClass, Placement};
use super::super::sharing::Sharing;
use super::super::Device;
use crate::release;
use ash::vk;

const MAX_BYTES: u64 = 64 << 20;
const MAX_BUFFER: u64 = 16 << 20;
const MAX_BUFFERS: usize = 16;

/// A pooled buffer can serve both submission paths on successive leases. Include the dedicated
/// staging family too: waiting for a submission does not transfer an exclusive buffer's owner.
fn buffer_info<'a>(
    len: u64,
    graphics: Sharing<'a>,
    staging: Option<Sharing<'a>>,
) -> vk::BufferCreateInfo<'a> {
    staging.unwrap_or(graphics).buffer(
        vk::BufferCreateInfo::default()
            .size(len)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
    )
}

/// Idle buffers with their usable capacity and allocation size. The latter bounds retained
/// placements, including the device's alignment padding, rather than only the bytes copied.
struct Idle<T> {
    buffers: Vec<(T, u64, u64)>,
    bytes: u64,
}

impl<T> Default for Idle<T> {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
            bytes: 0,
        }
    }
}

impl<T> Idle<T> {
    fn take(&mut self, len: u64) -> Option<T> {
        let index = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, (_, capacity, _))| *capacity >= len)
            .min_by_key(|(_, (_, capacity, _))| *capacity)
            .map(|(index, _)| index)?;
        let (buffer, _, allocation) = self.buffers.swap_remove(index);
        self.bytes -= allocation;
        Some(buffer)
    }

    fn give(&mut self, buffer: T, capacity: u64, allocation: u64) -> Option<T> {
        if allocation > MAX_BUFFER
            || self.buffers.len() >= MAX_BUFFERS
            || allocation > MAX_BYTES - self.bytes
        {
            return Some(buffer);
        }
        self.bytes += allocation;
        self.buffers.push((buffer, capacity, allocation));
        None
    }
}

struct Buffer {
    raw: vk::Buffer,
    placement: Placement,
    capacity: u64,
}

impl Buffer {
    /// Release a buffer whose synchronous submission completed, or which was never submitted.
    fn destroy(self, device: &ash::Device, memory: &mut Memory) {
        // SAFETY: no pending submission uses this buffer.
        unsafe { release::object(device, self.raw) };
        memory.free(device, &self.placement);
    }
}

/// The device's idle texture staging buffers. They hold no device reference.
#[derive(Default)]
pub(crate) struct StagingPool {
    idle: Idle<Buffer>,
}

impl StagingPool {
    /// Release the idle buffers before the device's allocator is destroyed; the device is idle.
    pub(crate) fn destroy(&mut self, device: &ash::Device, memory: &mut Memory) {
        for (buffer, _, _) in self.idle.buffers.drain(..) {
            buffer.destroy(device, memory);
        }
        self.idle.bytes = 0;
    }
}

/// An exclusive staging lease. Only `recycle`, after a successful synchronous wait and all CPU
/// accesses, lets the buffer be reused; error paths release it as before.
pub(super) struct Staging<'a> {
    device: &'a Device,
    buffer: Option<Buffer>,
    len: usize,
    reusable: bool,
}

impl Staging<'_> {
    pub(super) fn new(device: &Device, len: u64) -> Result<Staging<'_>, Error> {
        let size = usize::try_from(len).map_err(|_| Error::OutOfMemory(MemoryKind::Host))?;
        let shared = device.shared();
        let idle = lock(&shared.textures.staging).idle.take(len);
        if let Some(buffer) = idle {
            return Ok(Staging {
                device,
                buffer: Some(buffer),
                len: size,
                reusable: false,
            });
        }
        let raw_device = shared.vk();
        let info = buffer_info(len, shared.sharing(), device.staging_sharing());
        // SAFETY: the device is live and `info` valid.
        let raw = unsafe { raw_device.create_buffer(&info, None) }.map_err(vulkan)?;
        // SAFETY: `raw` is a live buffer of this device.
        let requirements = unsafe { raw_device.get_buffer_memory_requirements(raw) };
        let placement = match crate::memory::allocate(
            &shared.memory,
            raw_device,
            &requirements,
            MemoryClass::Shared,
        ) {
            Ok(placement) => placement,
            Err(error) => {
                // SAFETY: the buffer is unused.
                unsafe { release::object(raw_device, raw) };
                return Err(error);
            }
        };
        let buffer = Buffer {
            raw,
            placement,
            capacity: len,
        };
        let bound = if buffer.placement.mapped.is_none() {
            Err(Error::OutOfMemory(MemoryKind::Host))
        } else {
            // SAFETY: the placement meets the buffer's requirements and is not bound elsewhere.
            unsafe {
                raw_device.bind_buffer_memory(raw, buffer.placement.memory, buffer.placement.offset)
            }
            .map_err(vulkan)
        };
        if let Err(error) = bound {
            buffer.destroy(raw_device, &mut lock(&shared.memory));
            return Err(error);
        }
        Ok(Staging {
            device,
            buffer: Some(buffer),
            len: size,
            reusable: false,
        })
    }

    fn buffer(&self) -> &Buffer {
        self.buffer
            .as_ref()
            .unwrap_or_else(|| unreachable!("a live staging lease holds its buffer"))
    }

    pub(super) fn raw(&self) -> vk::Buffer {
        self.buffer().raw
    }

    pub(super) fn bytes(&mut self) -> &mut [u8] {
        let Some(mapped) = self.buffer().placement.mapped else {
            unreachable!("checked when the staging buffer was made");
        };
        // SAFETY: this lease exclusively owns at least `len` host-coherent mapped bytes, and
        // callers wait for every submission before accessing them on the CPU again.
        unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr(), self.len) }
    }

    /// Return this lease only after every submission using it successfully completed. Consuming
    /// it prevents another submission or CPU access after the buffer becomes reusable.
    pub(super) fn recycle(mut self) {
        self.reusable = true;
    }
}

impl Drop for Staging<'_> {
    fn drop(&mut self) {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let Some(buffer) = self.buffer.take() else {
            return;
        };
        let shared = self.device.shared();
        let released = if self.reusable && !self.device.is_lost() {
            let (capacity, allocation) = (buffer.capacity, buffer.placement.size());
            lock(&shared.textures.staging)
                .idle
                .give(buffer, capacity, allocation)
        } else {
            Some(buffer)
        };
        // Vulkan destruction and allocator work happen outside the pool lock.
        if let Some(buffer) = released {
            buffer.destroy(shared.vk(), &mut lock(&shared.memory));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooled_buffers_include_the_dedicated_transfer_family() {
        let graphics = [2];
        let both = [2, 5];
        let info = buffer_info(4096, Sharing::new(&graphics), Some(Sharing::new(&both)));
        assert!(info.sharing_mode == vk::SharingMode::CONCURRENT);
        assert_eq!(info.queue_family_index_count, 2);
        assert_eq!(info.p_queue_family_indices, both.as_ptr());
        assert_eq!(info.size, 4096);
        assert!(info.usage.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert!(info.usage.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn a_device_without_a_transfer_queue_keeps_its_graphics_sharing() {
        let families = [2, 3];
        let info = buffer_info(4096, Sharing::new(&families), None);
        assert!(info.sharing_mode == vk::SharingMode::CONCURRENT);
        assert_eq!(info.queue_family_index_count, 2);
        assert_eq!(info.p_queue_family_indices, families.as_ptr());
        let family = [2];
        let info = buffer_info(4096, Sharing::new(&family), None);
        assert!(info.sharing_mode == vk::SharingMode::EXCLUSIVE);
    }

    #[test]
    fn leases_are_exclusive_and_choose_the_smallest_sufficient_buffer() {
        let mut idle = Idle::default();
        assert_eq!(idle.give(1, 4096, 8192), None);
        assert_eq!(idle.give(2, 1024, 2048), None);
        assert_eq!(idle.take(512), Some(2));
        assert_eq!(idle.take(4097), None);
        assert_eq!(idle.take(4096), Some(1));
        assert_eq!(idle.take(1), None);
        assert_eq!(idle.bytes, 0);
        assert_eq!(idle.give(2, 1024, 2048), None);
        assert_eq!(idle.take(1024), Some(2));
    }

    #[test]
    fn idle_allocations_are_bounded_by_bytes_and_count() {
        let mut idle = Idle::default();
        assert_eq!(idle.give(0, MAX_BUFFER, MAX_BUFFER + 1), Some(0));
        for id in 0..4 {
            assert_eq!(idle.give(id, 1, MAX_BUFFER), None);
        }
        assert_eq!(idle.bytes, MAX_BYTES);
        assert_eq!(idle.give(5, 1, 1), Some(5));
        assert!(idle.take(1).is_some());
        assert_eq!(idle.give(6, 1, MAX_BUFFER), None);
        let mut small = Idle::default();
        for id in 0..MAX_BUFFERS {
            assert_eq!(small.give(id, 1, 1), None);
        }
        assert_eq!(small.give(MAX_BUFFERS, 1, 1), Some(MAX_BUFFERS));
    }

    #[cfg(any(m2v_gpu, m2v_cpu_icd))]
    #[test]
    fn completed_copies_reuse_the_buffer_with_exclusive_mapped_bytes() {
        use crate::{DeviceKind, DeviceOptions};

        let device = Device::system_default_with(&DeviceOptions {
            kinds: if cfg!(m2v_cpu_icd) {
                vec![DeviceKind::Cpu]
            } else {
                vec![
                    DeviceKind::Discrete,
                    DeviceKind::Integrated,
                    DeviceKind::Virtual,
                ]
            },
            ..DeviceOptions::default()
        })
        .unwrap();
        let unused = Staging::new(&device, 4096).unwrap();
        let mut filled = Staging::new(&device, 4096).unwrap();
        let buffer = filled.raw();
        assert_ne!(unused.raw(), buffer);
        device
            .submit_internal(|raw, command_buffer| {
                // SAFETY: this exclusively leased buffer is live, has TRANSFER_DST usage and
                // holds 4096 bytes. The submission is waited for before reading the mapping.
                unsafe { raw.cmd_fill_buffer(command_buffer, buffer, 0, 4096, 0xa1b2c3d4) };
                super::super::transfer_barrier(raw, command_buffer, false);
            })
            .unwrap();
        assert!(filled
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .all(|word| word == &[0xd4, 0xc3, 0xb2, 0xa1]));
        assert!(lock(&device.shared().textures.staging)
            .idle
            .buffers
            .is_empty());
        drop(unused);
        filled.recycle();
        let mut reused = Staging::new(&device, 1024).unwrap();
        assert_eq!(reused.raw(), buffer);
        assert_eq!(reused.bytes().len(), 1024);
        assert_eq!(&reused.bytes()[..4], &[0xd4, 0xc3, 0xb2, 0xa1]);
        let concurrent = Staging::new(&device, 1024).unwrap();
        assert_ne!(concurrent.raw(), buffer);
        drop(concurrent);
        reused.recycle();
        assert_eq!(
            lock(&device.shared().textures.staging).idle.buffers.len(),
            1
        );
    }
}
