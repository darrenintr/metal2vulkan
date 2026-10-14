//! Consecutive batches' submissions to one `VkQueue`, made as one `vkQueueSubmit` of one
//! `VkSubmitInfo` (a runtime choice: Metal has no submission of its own to share).
//!
//! **Why.** A native driver's submission is a kernel call whose cost does not follow its work:
//! RADV's (Mesa 26.2.4, Linux 7.3, an RX 7600) is one `DRM_IOCTL_AMDGPU_CS` that validates every
//! imported buffer, 0.3 ms with 250 imports and 1.4 ms with 1,150, measured from a macOS 26
//! guest's window animations, which commit 1,000 to 1,700 command buffers a second. Batches the
//! scheduler finds ready one after another in a pass go as one submission: their waits, their
//! command buffers in order, their signals.
//!
//! **Why it is the same.** Each batch's waits are satisfied by signals already submitted (the
//! scheduler's order), never by a later batch, so making all of a run's waits the first
//! batch's only delays work, and never waits for the run itself. The command buffers keep their
//! order in the submission, and a batch's barriers order it after the commands before it in
//! submission order, as across submissions. A timeline signalled more than once keeps the
//! highest value, which every wait for a lower one sees too (a timeline's values rise). Where a
//! batch would wait for what an earlier batch of the run signals, or two would wait for or
//! signal the same binary semaphore, the run ends before it.
//!
//! **Where it does not.** A batch on MoltenVK, which relays failures per submission
//! (`device/relay.rs`), a batch with a separate first-work submission, and every other use of
//! the queue (a sparse bind, the program's) end the run first. A batch whose submission takes a
//! fence ends the run with that fence. A run never outlives the scheduler's lock: the pass that
//! made it submits it before it lets go ([`Coalesced::flush`]). A failed submission of a run
//! whose batches were taken as submitted loses the device, as Vulkan leaves such a queue.

use crate::device::relay::Relayed;
use crate::device::Device;
use crate::exit_guard::{self, Call};
use ash::vk;
use std::sync::Arc;

/// The most batches one submission takes.
const RUN_LIMIT: usize = 16;

/// One batch's part of a submission.
pub(crate) struct Part {
    /// Its waits: a semaphore, a value (0 for a binary one) and the stages it holds.
    pub(crate) waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags)>,
    /// Its command buffers, in order.
    pub(crate) buffers: Vec<vk::CommandBuffer>,
    /// Its signals: a semaphore and a value (0 for a binary one).
    pub(crate) signals: Vec<(vk::Semaphore, u64)>,
}

/// The run being gathered ([`Coalesced::add`]).
#[derive(Default)]
pub(crate) struct Coalesced {
    queue: Option<usize>,
    waits: Vec<(vk::Semaphore, u64, vk::PipelineStageFlags)>,
    buffers: Vec<vk::CommandBuffer>,
    signals: Vec<(vk::Semaphore, u64)>,
    batches: usize,
    relayed: Option<Arc<Relayed>>,
    device: Option<Device>,
}

impl Coalesced {
    /// Whether `part` cannot join the run: it waits for what the run signals, or it would wait
    /// for or signal a binary semaphore the run already does.
    fn conflicts(&self, part: &Part) -> bool {
        let signals = |s: vk::Semaphore| self.signals.iter().find(|(x, _)| *x == s).map(|e| e.1);
        let waits = |s: vk::Semaphore| self.waits.iter().find(|(x, ..)| *x == s).map(|e| e.1);
        part.waits
            .iter()
            .any(|&(s, v, _)| signals(s).is_some() || waits(s).is_some_and(|w| w == 0 || v == 0))
            || part
                .signals
                .iter()
                .any(|&(s, v)| signals(s).is_some_and(|w| w == 0 || v == 0))
    }

    /// Take `part`, `device`'s batch for `VkQueue` `queue` whose failures `relayed` hears, into
    /// the run; a run it cannot join is submitted first, and one it ends with its `fence` after.
    /// `Err` for a failed submission of an earlier run, which lost the device.
    pub(crate) fn add(
        &mut self,
        device: &Device,
        queue: usize,
        part: Part,
        fence: Option<vk::Fence>,
        relayed: &Arc<Relayed>,
    ) -> Result<(), vk::Result> {
        if self.batches > 0
            && (self.queue != Some(queue) || self.batches >= RUN_LIMIT || self.conflicts(&part))
        {
            self.flush(None)?;
        }
        self.queue = Some(queue);
        self.device.get_or_insert_with(|| device.clone());
        self.relayed = Some(Arc::clone(relayed));
        for (s, v, stages) in part.waits {
            match self.waits.iter_mut().find(|(x, ..)| *x == s) {
                Some(wait) => {
                    wait.1 = wait.1.max(v);
                    wait.2 |= stages;
                }
                None => self.waits.push((s, v, stages)),
            }
        }
        self.buffers.extend(part.buffers);
        for (s, v) in part.signals {
            match self.signals.iter_mut().find(|(x, _)| *x == s) {
                Some(signal) => signal.1 = signal.1.max(v),
                None => self.signals.push((s, v)),
            }
        }
        self.batches += 1;
        if fence.is_some() {
            self.flush(fence)?;
        }
        Ok(())
    }

    /// Submit the run, if any, with `fence`, under the queue's lock, inside a section of the
    /// exit guard; once the process is exiting it is left, as everything held is. A failed
    /// submission loses the device.
    pub(crate) fn flush(&mut self, fence: Option<vk::Fence>) -> Result<(), vk::Result> {
        let taken = std::mem::take(self);
        let (Some(queue), Some(device), Some(relayed)) = (taken.queue, taken.device, taken.relayed)
        else {
            return Ok(());
        };
        if taken.batches == 0 {
            return Ok(());
        }
        let Some(_busy) = exit_guard::enter() else {
            return Ok(());
        };
        let shared = device.shared();
        let raw = shared.vk();
        let wait_semaphores: Vec<vk::Semaphore> = taken.waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<u64> = taken.waits.iter().map(|w| w.1).collect();
        let wait_stages: Vec<vk::PipelineStageFlags> = taken.waits.iter().map(|w| w.2).collect();
        let signal_semaphores: Vec<vk::Semaphore> = taken.signals.iter().map(|s| s.0).collect();
        let signal_values: Vec<u64> = taken.signals.iter().map(|s| s.1).collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&taken.buffers)
            .signal_semaphores(&signal_semaphores)
            .push_next(&mut timeline);
        // SAFETY: as each batch's submission (`command_buffer/submit.rs`): every command buffer
        // is executable and submitted once, every wait is for a signal already submitted (none
        // the run's own: `conflicts`), every timeline signal is above its value, every binary
        // one of an unsignaled semaphore with no signal pending (one per run: `conflicts`); the
        // fence is unsignaled and in no pending submission; the queue is used under its lock.
        let submitted = shared.submit_logged(queue, fence, &relayed, |vk_queue| unsafe {
            raw.queue_submit(
                vk_queue,
                std::slice::from_ref(&info),
                fence.unwrap_or_default(),
            )
        });
        exit_guard::called(Call::QueueSubmit);
        // A failed one's fence is its batch's to release (`command_buffer/submit.rs`).
        if submitted.is_err() {
            device.mark_lost();
        }
        submitted
    }
}
