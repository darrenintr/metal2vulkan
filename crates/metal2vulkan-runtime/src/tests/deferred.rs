//! `scheduler::defer_submissions`: commits a thread defers are submitted together by a later pass,
//! in commit order, and no wait waits on them: the deferring thread's own wait, another thread's
//! wait, and a commit from a thread that does not defer each submit them first.
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    defer_submissions, CommandBuffer, CommandBufferStatus, CommandQueue, Device, DeviceKind,
    DeviceOptions, ResourceOptions,
};
use std::time::Duration;

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU.
fn device() -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// A command buffer of one fill of `buffer`'s first 4 KiB with `value`.
fn fill(queue: &CommandQueue, buffer: &crate::Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer
}

/// The first byte of a shared buffer.
fn first(buffer: &crate::Buffer) -> u8 {
    // SAFETY: a shared buffer's contents, at least 4 KiB, read after its work completed.
    unsafe { *buffer.contents().unwrap().cast::<u8>().as_ptr() }
}

/// A run of deferred commits completes, in order, once the deferring thread waits for the last:
/// the last fill's value is what the buffer holds
#[test]
fn deferred_commits_complete_in_order_at_the_deferring_threads_wait() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let command_buffers: Vec<_> = (1..=10).map(|value| fill(&queue, &buffer, value)).collect();
    defer_submissions(true);
    for command_buffer in &command_buffers {
        command_buffer.commit().unwrap();
    }
    command_buffers[9].wait_until_completed_for(WAIT).unwrap();
    defer_submissions(false);
    for command_buffer in &command_buffers {
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    assert_eq!(first(&buffer), 10);
}

/// A commit one thread deferred, and never submits itself, completes when another thread waits
/// for it
#[test]
fn another_threads_wait_submits_a_deferred_commit() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let command_buffer = fill(&queue, &buffer, 7);
    let committer = {
        let command_buffer = command_buffer.clone();
        std::thread::spawn(move || {
            defer_submissions(true);
            command_buffer.commit().unwrap();
            // Leaves deferring without a flush of its own: the thread-local list goes with it.
            std::thread::sleep(Duration::from_millis(200));
        })
    };
    std::thread::sleep(Duration::from_millis(20));
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(first(&buffer), 7);
    committer.join().unwrap();
}

/// A commit from a thread that does not defer submits the deferred commits before it: the later
/// commit's fill lands last
#[test]
fn a_commit_that_does_not_defer_submits_the_deferred_ones_first() {
    let device = device();
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::shared()).unwrap();
    let early = fill(&queue, &buffer, 1);
    let late = fill(&queue, &buffer, 2);
    {
        let early = early.clone();
        std::thread::spawn(move || {
            defer_submissions(true);
            early.commit().unwrap();
        })
        .join()
        .unwrap();
    }
    late.commit().unwrap();
    late.wait_until_completed_for(WAIT).unwrap();
    early.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(first(&buffer), 2);
}
