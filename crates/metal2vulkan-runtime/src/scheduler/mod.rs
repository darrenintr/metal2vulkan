//! The device's scheduler: one thread per device that submits the committed command buffers'
//! batches to the device's `VkQueue`s.
//!
//! - **Submitted by the commit**. `commit` makes the thread's pass itself, on the committing
//!   thread, under the scheduler's lock: the batches ready then go to the `VkQueue` before it
//!   returns, in the same order, with no wake of the thread on their path. The thread runs what
//!   that pass leaves (a host step, a submission check, a wait for an event the host raises later),
//!   as before.
//!
//! - **The order** ([`Order`]). Each logical queue's batches wait in commit order; the head
//!   committed first among those ready goes next. A batch is ready once every event it waits for
//!   is reached by a signal already submitted or by the host ([`EventTimeline`]), so no `VkQueue`
//!   ever holds a wait for a signal that is not submitted: logical queues sharing a `VkQueue`
//!   cannot deadlock through the order they wait for each other in.
//! - **Binary semaphores** ([`BinarySignals`]). A batch that waits for one of the caller's binary
//!   semaphores is ready once every signal of it that a command buffer of this device committed
//!   before is submitted; a signal from the caller was submitted before the commit.
//! - **Tracking** ([`Tracker`]). Each batch is planned as it is submitted, against the device's
//!   hazard state, in this order: the order Metal follows across queues. A batch held for its host
//!   step or its checks keeps its place among the tracked accesses: a later commit's batch on
//!   another queue that conflicts with it waits behind it, its own host step and checks too
//!   ([`Order::behind`]).
//! - **Many logical queues on few `VkQueue`s.** A logical queue keeps one `VkQueue` (round robin
//!   at creation); each `VkQueue` signals its own timeline semaphore with every batch, which is
//!   the [`Point`] the tracker and the completion thread know the batch by.
//! - **Scheduled.** A command buffer is scheduled once its last batch is submitted (its every
//!   wait was satisfied): the thread that submitted it (this one, or the committing one) posts
//!   its transition to `Scheduled` to its queue's context ([`contexts`](super::contexts)), which
//!   runs its scheduled handlers. No submission runs a scheduled handler, so none holds back a
//!   submission.
//! - **Submission handlers** (a runtime extension). The thread that submitted a command buffer's
//!   last batch runs its submission handlers, in submission order among the command buffers its
//!   pass submitted, once it let go of the lock and before it lets go of the command buffers: in
//!   `commit`, before it returns; on this thread, before its next pass, which they hold back. The
//!   command buffer's transition to `Scheduled` waits for them.
//! - **Host steps**. A batch that executes commands the GPU may write
//!   (`CommandBufferShared::needs_host_step`) is held, once otherwise ready, until every
//!   `VkQueue` completed what was submitted to it by then (its gate, a snapshot of the queues'
//!   values): only then are the records the work before it wrote known. The thread then records
//!   the batch's segments (`CommandBufferShared::host_step`) without its lock, each step a
//!   section of the exit guard, and submits the batch in its order. Meanwhile it submits other
//!   ready batches; the completion thread wakes it as batches complete while a gate is pending.
//! - **Submission checks** (`command_buffer/checks.rs`). A batch of a command buffer with a check
//!   is held, once otherwise ready and its host step recorded, until the checks of Metal's
//!   submissions in it ran (`CommandBufferShared::run_checks`), which the thread runs as it runs
//!   host steps: without its lock, each pass a section of the exit guard. Its checks wait, besides,
//!   until every event value it waits for was reached on the device or by the host: a value only a
//!   submitted GPU signal reaches is held for until that signal landed, as Metal submits (and
//!   checks) only then. The completion thread sees each finished batch's event signals before it
//!   wakes the thread, as for a gate. A command buffer without a check costs one flag per batch.
//! - **Waits are bounded.** The thread wakes on a commit or a host signal, and at least every
//!   [`WAIT_SLICE`], so a stop is seen promptly.
//! - **Exit.** Each submission is a section of the exit guard ([`exit_guard`]); once the process
//!   is exiting, the thread ends, and what it holds stays held.
//! - **Let go without the lock.** A submitted batch's hold of its command buffer goes after the
//!   lock: it may be the last reference, and the command buffer's device may go with it, whose
//!   drop stops this thread.
//!
//! Locks are taken in one order: the scheduler's state, then a command buffer's, then an event's
//! or a `VkQueue`'s, then the handler threads' state, then the completion thread's list. No call
//! made under the scheduler's lock takes it again.
//!
//! [`EventTimeline`]: super::event_timeline::EventTimeline
//! [`Point`]: super::tracker::Point

pub(super) mod binary;
pub(crate) mod coalesce;
pub(super) mod order;

use super::command_buffer::gate::Hold;
use super::command_buffer::CommandBufferShared;
use super::device::lock;
use super::error::{Error, MemoryKind};
use super::event::EventShared;
use super::event_timeline::WaitFor;
use super::exit_guard;
use super::replay::Replay;
use super::resource::ResourceKey;
use super::tracker::{self, Footprint, Tracker};
use ash::vk;
use binary::{BinarySignals, Needs};
use order::{Behind, Order};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;
use std::sync::atomic::{fence, AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::JoinHandle;
use std::time::Duration;

/// The passes `commit` makes at most on the committing thread, running between them the
/// submission checks a pass found ready (`Scheduler::commit`); what is left then is the thread's.
const INLINE_CHECK_PASSES: u32 = 4;

/// How long the thread sleeps at most without looking for work.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// One device's scheduler thread.
pub(crate) struct Scheduler {
    shared: Arc<SchedulerShared>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

struct SchedulerShared {
    state: Mutex<State>,
    wake: Condvar,
    /// Commits not yet taken into `state`, in commit order (`Scheduler::commit`): a commit that
    /// finds the state's lock held leaves its batches here rather than waiting behind the
    /// holder, which may be in a `vkQueueSubmit`. Whoever next holds the lock and passes takes
    /// them in order first ([`take_incoming`]).
    incoming: Mutex<std::collections::VecDeque<Incoming>>,
    /// `incoming` may hold commits: the thread does not wait while it is set.
    pending: AtomicBool,
    /// Since when `incoming` holds commits a deferring thread left there without a pass
    /// ([`defer_submissions`]): the thread takes them once they are [`DEFER_LIMIT`] old, should
    /// no pass take them before.
    deferred_since: Mutex<Option<std::time::Instant>>,
    /// A batch waits at its gate for work to complete: the completion thread wakes the thread
    /// as batches complete.
    gated: AtomicBool,
}

/// What the scheduler submits from, and the device state it submits against.
pub(crate) struct State {
    order: Order<Held>,
    /// The device's hazard state, in submission order.
    pub(crate) tracker: Tracker<ResourceKey>,
    /// Per `VkQueue`: the timeline value of the last batch submitted there.
    pub(crate) values: Vec<u64>,
    /// The binary semaphore signals the device's command buffers committed and submitted.
    binary: BinarySignals<vk::Semaphore>,
    /// A commit or a host signal came since the thread last looked.
    dirty: bool,
    stopping: bool,
    /// The textures whose decoded twins have CPU decodes queued or running, each with how many
    /// (`texture/twin_decode.rs`): a batch whose footprint names one waits for them.
    decodes: BTreeMap<ResourceKey, usize>,
    /// How many of `decodes` are of untracked textures, whose uses no footprint names: every
    /// batch waits while one is left.
    untracked_decodes: usize,
    /// The batches a pass submitted one after another, made as one submission
    /// ([`coalesce`]): submitted before the pass ends.
    pub(crate) coalesced: coalesce::Coalesced,
}

/// A commit left for the next pass ([`SchedulerShared::incoming`]).
struct Incoming {
    buffer: Arc<CommandBufferShared>,
    queue: u64,
    batches: Vec<BatchSync>,
}

/// Takes the commits left in `incoming` into the order, in commit order, as `commit` would
/// have under the lock. `pending` is cleared first, so a commit left after it is seen again.
fn take_incoming(shared: &SchedulerShared, state: &mut State) {
    shared.pending.store(false, Ordering::SeqCst);
    *lock(&shared.deferred_since) = None;
    let commits: Vec<Incoming> = lock(&shared.incoming).drain(..).collect();
    for Incoming {
        buffer,
        queue,
        batches,
    } in commits
    {
        hold_batches(state, &buffer, queue, batches);
    }
}

/// Holds `buffer`'s batches, committed on logical queue `queue`, in the order.
fn hold_batches(
    state: &mut State,
    buffer: &Arc<CommandBufferShared>,
    queue: u64,
    batches: Vec<BatchSync>,
) {
    let count = batches.len();
    let binary = &mut state.binary;
    let held: Vec<Held> = batches
        .into_iter()
        .enumerate()
        .map(|(index, batch)| Held {
            buffer: buffer.hold(),
            index,
            last: index + 1 == count,
            binary_needs: binary.commit(&batch.binary_waits, &batch.binary_signals),
            waits: batch.waits,
            binary_signals: batch.binary_signals,
            segments: batch.segments,
            gate: None,
            checked: batch.checked,
        })
        .collect();
    state.order.push(queue, held);
}

/// A committed batch not yet submitted.
struct Held {
    buffer: Hold<CommandBufferShared>,
    /// Its index among the command buffer's batches.
    index: usize,
    /// Whether it is the command buffer's last batch.
    last: bool,
    /// The events it waits for, with the values.
    waits: Vec<(Arc<EventShared>, u64)>,
    /// What its binary semaphore waits need submitted first.
    binary_needs: Needs<vk::Semaphore>,
    /// The binary semaphores it signals.
    binary_signals: Vec<vk::Semaphore>,
    /// It holds segments, so it may need a host step.
    segments: bool,
    /// For its host step: each `VkQueue`'s value of the last batch submitted there once the batch
    /// was otherwise ready, which the queue's completed value reaches before the step.
    gate: Option<Vec<u64>>,
    /// Its command buffer has a submission check, so it may wait for checks to run.
    checked: bool,
}

/// What a batch's host step waits for (`CommandBufferShared::step_waits`).
pub(crate) struct StepWaits {
    /// The work submitted before it to complete (its gate): the GPU may write what it reads.
    pub(crate) gpu: bool,
    /// A pipeline it records, still being made.
    pub(crate) pipelines: bool,
}

/// What one committed batch waits for and signals, as the scheduler orders it.
pub(crate) struct BatchSync {
    /// The events it waits for, with the values.
    pub(crate) waits: Vec<(Arc<EventShared>, u64)>,
    /// The caller's binary semaphores it waits for.
    pub(crate) binary_waits: Vec<vk::Semaphore>,
    /// The caller's binary semaphores it signals.
    pub(crate) binary_signals: Vec<vk::Semaphore>,
    /// It holds segments, so it may need a host step.
    pub(crate) segments: bool,
    /// Its command buffer has a submission check, so it may wait for checks to run.
    pub(crate) checked: bool,
}

impl Scheduler {
    /// Start the thread of a device with `queues` `VkQueue`s; refused when the host cannot start
    /// one.
    pub(crate) fn start(queues: usize) -> Result<Scheduler, Error> {
        let shared = Arc::new(SchedulerShared {
            state: Mutex::new(State {
                order: Order::default(),
                tracker: Tracker::default(),
                values: vec![0; queues],
                binary: BinarySignals::default(),
                dirty: false,
                stopping: false,
                decodes: BTreeMap::new(),
                untracked_decodes: 0,
                coalesced: coalesce::Coalesced::default(),
            }),
            wake: Condvar::new(),
            gated: AtomicBool::new(false),
            incoming: Mutex::new(std::collections::VecDeque::new()),
            pending: AtomicBool::new(false),
            deferred_since: Mutex::new(None),
        });
        let thread = {
            let shared = Arc::clone(&shared);
            exit_guard::spawn(
                std::thread::Builder::new().name("metal2vulkan-scheduler".into()),
                move || run(&shared),
            )
            .map_err(|_| Error::OutOfMemory(MemoryKind::Host))?
        };
        Ok(Scheduler {
            shared,
            thread: Mutex::new(Some(thread)),
        })
    }

    /// Hold the batches of `buffer`, committed now on logical queue `queue`: what each batch
    /// waits for and signals, in order. Then submit, on the committing thread, every batch that
    /// is ready now, as the thread's pass would: a commit whose batches wait for nothing reaches
    /// the `VkQueue` before `commit` returns, without the thread's wake on its path. A host step or
    /// a submission check found ready, and the process's exit, are left to the thread, which this
    /// wakes for them.
    pub(crate) fn commit(
        &self,
        buffer: &Arc<CommandBufferShared>,
        queue: u64,
        batches: Vec<BatchSync>,
    ) {
        // In commit order through `incoming`: a commit that finds the lock held (the thread may
        // be in a `vkQueueSubmit`) leaves its batches there and wakes the thread, which takes
        // them first in its next pass, rather than waiting behind the submission.
        lock(&self.shared.incoming).push_back(Incoming {
            buffer: Arc::clone(buffer),
            queue,
            batches,
        });
        if DEFERRING.with(Cell::get) {
            defer(&self.shared);
            return;
        }
        pass_now(&self.shared);
    }

    /// What makes the thread look for work again from any thread: a pipeline a held batch's
    /// host step waits for was made (`render/pending.rs`). It holds the thread's state weakly,
    /// so a making it waits on keeps nothing of the scheduler alive.
    pub(crate) fn waker(&self) -> impl Fn() + Clone + Send + 'static {
        let shared = Arc::downgrade(&self.shared);
        move || {
            if let Some(shared) = shared.upgrade() {
                lock(&shared.state).dirty = true;
                shared.wake.notify_all();
            }
        }
    }

    /// Whether every batch committed so far was submitted: none is held for an event, a binary
    /// semaphore, a host step or a check (`texture/storage_form.rs`).
    pub(crate) fn holds_none(&self) -> bool {
        let _ = flush(&self.shared);
        let state = lock(&self.shared.state);
        state.order.is_empty() && lock(&self.shared.incoming).is_empty()
    }

    /// A CPU decode into the decoded twin of the resource `key` was queued: every batch whose
    /// footprint names it waits at its submission until [`decode_finished`](Self::decode_finished)
    /// says so of every one queued; where it is not `tracked` (its uses not recorded), every
    /// batch. A batch already submitted is not waited for.
    pub(crate) fn decode_queued(&self, key: ResourceKey, tracked: bool) {
        let mut state = lock(&self.shared.state);
        *state.decodes.entry(key).or_default() += 1;
        if !tracked {
            state.untracked_decodes += 1;
        }
    }

    /// One CPU decode into the decoded twin of `key` (`tracked` as it was queued) finished: the
    /// batches waiting for its decodes go once none is left.
    pub(crate) fn decode_finished(&self, key: ResourceKey, tracked: bool) {
        let mut state = lock(&self.shared.state);
        if let Some(count) = state.decodes.get_mut(&key) {
            *count -= 1;
            if *count == 0 {
                state.decodes.remove(&key);
            }
        }
        if !tracked {
            state.untracked_decodes -= 1;
        }
        state.dirty = true;
        drop(state);
        self.shared.wake.notify_all();
    }

    /// Whether this thread is the scheduler's (a submission check runs on it).
    pub(crate) fn is_current(&self) -> bool {
        lock(&self.thread)
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id())
    }

    /// Look for work again: an event's value rose.
    pub(crate) fn wake(&self) {
        lock(&self.shared.state).dirty = true;
        self.shared.wake.notify_all();
    }

    /// Batches completed, their queues' completed values and their event signals published: look
    /// for work again when a batch waits at its gate or for a GPU signal to land. The flag is read
    /// after a fence, as the thread sets it before it reads the completed values or the events,
    /// so one of the two sees the other.
    pub(crate) fn completions(&self) {
        fence(Ordering::SeqCst);
        if self.shared.gated.load(Ordering::SeqCst) {
            self.wake();
        }
    }

    /// Ask the thread to end, and wait for it unless this is the thread. The device is being
    /// dropped, so nothing is held: every held command buffer holds its device.
    pub(crate) fn stop(&self) {
        lock(&self.shared.state).stopping = true;
        self.shared.wake.notify_all();
        let mut thread = lock(&self.thread);
        let own = thread
            .as_ref()
            .is_some_and(|handle| handle.thread().id() == std::thread::current().id());
        if !own {
            if let Some(handle) = thread.take() {
                let _ = handle.join();
            }
        }
    }
}

thread_local! {
    /// Whether this thread defers its commits' submissions ([`defer_submissions`]).
    static DEFERRING: Cell<bool> = const { Cell::new(false) };
    /// The schedulers this thread left commits with since its last flush.
    static DEFERRED_TO: RefCell<Vec<Weak<SchedulerShared>>> = const { RefCell::new(Vec::new()) };
}

/// The commits a thread defers in a row before it submits them anyway.
const DEFER_COUNT: usize = 32;

/// How long deferred commits wait at most for a pass before the thread takes them: a deferring
/// thread that waits on something its own deferred work must end bears at most this much.
const DEFER_LIMIT: Duration = Duration::from_millis(1);

/// Static registry of the schedulers holding deferred commits, for a flush from any thread.
static WITH_DEFERRED: Mutex<Vec<Weak<SchedulerShared>>> = Mutex::new(Vec::new());

/// `WITH_DEFERRED` may name a scheduler: a flush with nothing deferred costs one atomic read.
static ANY_DEFERRED: AtomicBool = AtomicBool::new(false);

/// From now on, this thread's commits are submitted later, together, rather than each by its own
/// pass (`on`), or no more, which submits what it deferred. A deferred commit goes into the
/// scheduler's `incoming` in commit order, where any pass takes it, the thread's included: a
/// commit from a thread that does not defer, [`flush_deferred_submissions`], any wait for a command
/// buffer or an event, the deferring thread's `DEFER_COUNT`th commit in a row, or the scheduler's
/// own thread once the oldest is [`DEFER_LIMIT`] old. Commit order is kept across threads, and no
/// wait waits on deferred work for long.
pub fn defer_submissions(on: bool) {
    DEFERRING.with(|d| d.set(on));
    if !on {
        let _ = flush_deferred_submissions();
    }
}

/// Submit every commit deferred so far, on every device, by passes on this thread: `true` where
/// some scheduler held deferred commits, which a caller about to wait for something they may
/// end checks for again.
pub fn flush_deferred_submissions() -> bool {
    DEFER_RUN.with(|r| r.set(0));
    let mine: Vec<Weak<SchedulerShared>> = DEFERRED_TO.with(|d| d.borrow_mut().drain(..).collect());
    let others: Vec<Weak<SchedulerShared>> = if ANY_DEFERRED.swap(false, Ordering::SeqCst) {
        lock(&WITH_DEFERRED).drain(..).collect()
    } else {
        Vec::new()
    };
    let mut any = false;
    for shared in mine.iter().chain(&others).filter_map(Weak::upgrade) {
        any |= flush(&shared);
    }
    any
}

/// `shared`'s deferred commits go to its thread, if it holds any: whether it did. The thread
/// submits every commit waiting in `incoming` together in its next pass, and what arrives while
/// it is in a submission goes together in the one after: one submitter, and as few submissions
/// as the driver's time for each allows, where each deferring thread's own pass would submit
/// its few alone (on RADV a submission costs every imported buffer object it lists, 1.7 ms with
/// a thousand, measured under reims-vgpu on an RX 7600, Mesa 26.2.4).
fn flush(shared: &SchedulerShared) -> bool {
    let held = lock(&shared.deferred_since).is_some();
    if held {
        shared.pending.store(true, Ordering::SeqCst);
        shared.wake.notify_all();
    }
    held
}

/// Leave a commit already in `incoming` for a later pass (`defer_submissions`).
fn defer(shared: &Arc<SchedulerShared>) {
    let first = {
        let mut since = lock(&shared.deferred_since);
        let first = since.is_none();
        if first {
            *since = Some(std::time::Instant::now());
        }
        first
    };
    DEFERRED_TO.with(|d| {
        let mut d = d.borrow_mut();
        if !d.iter().any(|w| w.as_ptr() == Arc::as_ptr(shared)) {
            d.push(Arc::downgrade(shared));
        }
    });
    if first {
        lock(&WITH_DEFERRED).push(Arc::downgrade(shared));
        ANY_DEFERRED.store(true, Ordering::SeqCst);
        // The thread takes them at the limit, should no pass come first.
        shared.wake.notify_all();
    }
    let run = DEFER_RUN.with(|r| {
        let n = r.get() + 1;
        r.set(n);
        n
    });
    if run >= DEFER_COUNT {
        let _ = flush_deferred_submissions();
    }
}

thread_local! {
    /// The commits this thread deferred since its last flush.
    static DEFER_RUN: Cell<usize> = const { Cell::new(0) };
}

/// Submit, on this thread, every batch ready now, the commits in `incoming` taken first (the
/// second half of [`Scheduler::commit`]): the thread is left only a host step, a check it could
/// not run here, a held lock, or the exit, and woken for them.
fn pass_now(shared: &SchedulerShared) {
    shared.pending.store(true, Ordering::SeqCst);
    let mut state = match shared.state.try_lock() {
        Ok(state) => state,
        Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) => {
            shared.wake.notify_all();
            return;
        }
    };
    // A pass, and then, while it leaves the thread nothing but submission checks found
    // ready, those checks here, without the lock, and another pass: a checked command buffer
    // whose batches wait for nothing reaches the `VkQueue` before `commit` returns, as an
    // unchecked one does, without the thread's wake on its path (`INLINE_CHECK_PASSES`).
    let mut passes = 0;
    loop {
        passes += 1;
        take_incoming(&shared, &mut state);
        let (mut spent, mut settled) = (Spent::default(), Vec::new());
        let (mut steps, mut checks) = (Vec::new(), Vec::new());
        let mut handlers = Vec::new();
        let exiting = submit_ready(
            &mut state,
            (&mut spent, &mut settled),
            (&mut steps, &mut checks),
            &mut handlers,
            &shared.gated,
        )
        .is_err();
        let inline =
            !exiting && steps.is_empty() && !checks.is_empty() && passes < INLINE_CHECK_PASSES;
        // The pass was the thread's but for what it runs without the lock: the thread looks
        // again only for that, or for the exit (or for what woke it before, which this pass
        // saw).
        let thread = !inline && (exiting || !steps.is_empty() || !checks.is_empty());
        state.dirty = thread;
        drop(state);
        settle(settled);
        // On the committing thread, before `commit` returns.
        run_submission_handlers(handlers);
        // Without the lock, as the thread lets go of them: the thread finds the steps and
        // the checks again in its own pass.
        drop(spent);
        drop(steps);
        if inline {
            // As the thread runs them: without the lock, each a section of the exit guard;
            // once the process is exiting, what is left stays held.
            for (buffer, index) in &checks {
                let Some(_busy) = exit_guard::enter() else {
                    return;
                };
                buffer.run_checks(*index);
            }
            drop(checks);
            state = match shared.state.try_lock() {
                Ok(state) => state,
                Err(std::sync::TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
                Err(std::sync::TryLockError::WouldBlock) => {
                    // The holder passes after it: the checked batches are its to submit.
                    shared.pending.store(true, Ordering::SeqCst);
                    shared.wake.notify_all();
                    return;
                }
            };
            continue;
        }
        drop(checks);
        if thread {
            shared.wake.notify_all();
        }
        return;
    }
}

/// The thread's loop. It ends when the device stops it, or once the process is exiting
/// ([`exit_guard`]): what is held then stays held, as Metal's pending work stays pending.
fn run(shared: &SchedulerShared) {
    let stats = std::env::var_os("M2V_SCHED_STATS").is_some();
    let mut printed = std::time::Instant::now();
    loop {
        if stats && printed.elapsed() >= Duration::from_secs(1) {
            printed = std::time::Instant::now();
            if let Some(line) = coalesce::stats::line() {
                eprintln!("{line}");
            }
        }
        let mut state = lock(&shared.state);
        while !state.dirty && !state.stopping && !shared.pending.load(Ordering::SeqCst) {
            // Deferred commits no pass took within the limit are this thread's to take.
            let slice = match *lock(&shared.deferred_since) {
                Some(since) => match DEFER_LIMIT.checked_sub(since.elapsed()) {
                    Some(left) if !left.is_zero() => left,
                    _ => break,
                },
                None => WAIT_SLICE,
            };
            state = shared
                .wake
                .wait_timeout(state, slice)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        if state.stopping {
            return;
        }
        state.dirty = false;
        take_incoming(shared, &mut state);
        let (mut spent, mut settled) = (Spent::default(), Vec::new());
        let mut steps = Vec::new();
        let mut checks = Vec::new();
        let mut handlers = Vec::new();
        let exiting = submit_ready(
            &mut state,
            (&mut spent, &mut settled),
            (&mut steps, &mut checks),
            &mut handlers,
            &shared.gated,
        )
        .is_err();
        drop(state);
        settle(settled);
        // On this thread, before its next pass; none once the process is exiting, as the thread
        // then runs nothing more (their command buffers stay `Committed`).
        if exiting {
            drop(handlers);
        } else {
            run_submission_handlers(handlers);
        }
        // Without the lock: the last reference to a command buffer may be among them, and with
        // it its device, whose drop stops this thread.
        drop(spent);
        if exiting {
            return;
        }
        if steps.is_empty() && checks.is_empty() {
            continue;
        }
        // The submission checks, without the lock (they run the program's code), each pass a
        // section; once the process is exiting the thread ends at the next.
        for (buffer, index) in &checks {
            let Some(_busy) = exit_guard::enter() else {
                return;
            };
            buffer.run_checks(*index);
        }
        drop(checks);
        // The host steps, without the lock (they record, and may make what a dispatch makes),
        // each a section; once the process is exiting the thread ends at the next.
        for (buffer, index) in &steps {
            let Some(_busy) = exit_guard::enter() else {
                return;
            };
            buffer.host_step(*index);
            for &call in exit_guard::HOST_STEP {
                exit_guard::called(call);
            }
        }
        drop(steps);
        lock(&shared.state).dirty = true;
    }
}

/// What a pass of [`submit_ready`] lets go of, for the caller to drop without the lock: the
/// holds of the command buffers its batches came from, and the logs their first submissions
/// ended (`replay.rs`), either of which may hold a resource's last handle.
#[derive(Default)]
struct Spent {
    buffers: Vec<Hold<CommandBufferShared>>,
    logs: Vec<Arc<Replay>>,
}

/// Submit every batch that is ready, in order, until none is; each submission is a section of
/// the exit guard. A batch is published (or its failure posted) only after its section ended, so
/// the exit hook's registration at the section's end comes before anything the program can
/// learn of it (`exit_guard`). A batch's events are let go of before its command buffer
/// (`gate`), whose holds go to `spent`, for the caller to drop without the lock, as do the logs
/// a command buffer's first submission ended (`replay.rs`). A batch whose
/// host step may run now goes to `steps` (once), for the caller to record without the lock;
/// `gated` says whether one still waits at its gate. A batch whose submission checks may run
/// now goes to `checks` (once), for the caller to run without the lock. A command buffer whose
/// last batch was submitted, or that failed, and on which a fetched target's storage form rides
/// (`texture/storage_form.rs`) is settled and goes to `settled`, for the caller to announce
/// without the lock ([`settle`]). A command buffer whose last batch was submitted with
/// submission handlers goes to `handlers`, in submission order, for the caller to run them
/// without the lock ([`run_submission_handlers`]). `Err` once the process is exiting.
fn submit_ready(
    state: &mut State,
    (spent, settled): (&mut Spent, &mut Settled),
    (steps, checks): (&mut Steps, &mut Steps),
    handlers: &mut Handlers,
    gated: &AtomicBool,
) -> Result<(), Exiting> {
    loop {
        let busy = exit_guard::enter().ok_or(Exiting)?;
        // The heads that wait behind a batch committed before them that keeps its place: no step,
        // check or submission of theirs before it is submitted (`order.rs`).
        // Of those whose host step or checks may start (`gate_heads`, `check_heads`), and of the
        // ready ones in commit order until one goes (`Order::next`); `memo` holds what this pass
        // found.
        let mut memo = Behind::default();
        let behind: Vec<u64> = (state
            .order
            .heads_where(|held| held.segments || held.checked))
        .into_iter()
        .filter(|&sequence| is_behind(&state.order, &state.binary, &mut memo, sequence))
        .collect();
        let waiting = gate_heads(state, &behind, steps, gated);
        let landing = check_heads(state, &behind, checks, gated);
        let binary = &state.binary;
        let decodes = &state.decodes;
        let untracked_decodes = state.untracked_decodes;
        let behind = |order: &Order<Held>, sequence| is_behind(order, binary, &mut memo, sequence);
        let next = state.order.next(behind, |held| {
            if !binary.ready(&held.binary_needs) {
                return None;
            }
            // Its first submission waits while a rebind encodes encoders of it again.
            if held.index == 0 && held.buffer.reencode_pending() {
                return None;
            }
            // Held for a decoded twin's CPU decode it may read: any batch, while an untracked
            // texture's is left.
            if untracked_decodes > 0 || (!decodes.is_empty() && reads_decoding(held, decodes)) {
                return None;
            }
            // Held for its host step.
            if held.segments && held.buffer.needs_host_step(held.index) {
                return None;
            }
            // Held for its submission checks.
            if held.checked && held.buffer.needs_check(held.index) {
                return None;
            }
            event_waits(held)
        });
        let Some((queue, sequence, held, waits)) = next else {
            if !waiting && !landing {
                gated.store(false, Ordering::SeqCst);
            }
            // The run the pass gathered goes before the pass lets go of the lock. A failure lost
            // the device (`coalesce.rs`): the batches already published finish as it says.
            drop(busy);
            if state.coalesced.batches() > 0 {
                coalesce::stats::end(4);
            }
            let _ = state.coalesced.flush(None);
            return Ok(());
        };
        let submitted = held
            .buffer
            .submit_batch(held.index, held.last, &waits, state)
            .transpose();
        // A rebind marked it since it was found ready: back at its place, which it keeps, until
        // the rebind encoded its encoders again (`command_buffer/reencode.rs`), which wakes the
        // thread.
        let Some(submitted) = submitted else {
            drop(busy);
            state.order.put_back(queue, sequence, held);
            continue;
        };
        let Held {
            buffer,
            last,
            waits: events,
            binary_needs,
            binary_signals,
            ..
        } = held;
        drop(events);
        drop(busy);
        match submitted {
            Ok(submission) => {
                state.binary.submitted(&binary_needs, &binary_signals);
                if submission.runs_handlers() {
                    handlers.push(buffer.hold());
                }
                spent.logs.extend(buffer.publish(submission));
                if last && buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
            }
            Err(error) => {
                // The rest of the command buffer never runs (dropped here, before it can
                // finish).
                state.binary.dropped(&binary_needs);
                let rest = state
                    .order
                    .drop_head(queue, |next| next.buffer.same(&buffer));
                for held in &rest {
                    state.binary.dropped(&held.binary_needs);
                }
                spent
                    .buffers
                    .extend(rest.into_iter().map(|held| held.buffer));
                buffer.failed(error, &state.values);
                if buffer.settle_storage_forms() {
                    settled.push(buffer.hold());
                }
            }
        }
        spent.buffers.push(buffer);
    }
}

/// The process is exiting ([`exit_guard`]): the thread submits no more.
struct Exiting;

/// Whether `held` may read a texture of `decodes`: its footprint names one, or it has none it can
/// know before its submission (sparse work, a segment whose uses its host step finds).
fn reads_decoding(held: &Held, decodes: &BTreeMap<ResourceKey, usize>) -> bool {
    held.buffer
        .footprint(held.index)
        .is_none_or(|footprint| footprint.names_any(|key| decodes.contains_key(key)))
}

/// Wake what waits for the storage forms riding on the `settled` command buffers
/// (`texture/storage_form.rs`), with the scheduler's lock free: the device's switch lock is
/// taken before it, never under it.
fn settle(settled: Settled) {
    for buffer in &settled {
        buffer.device().storage_form_settled();
    }
}

/// The command buffers whose storage forms the scheduler settled, to announce without its lock.
type Settled = Vec<Hold<CommandBufferShared>>;

/// The command buffers whose last batch a pass submitted with submission handlers, in
/// submission order, for the submitting thread to run without the lock.
type Handlers = Vec<Hold<CommandBufferShared>>;

/// Run the submission handlers of `handlers`' command buffers on this thread, which holds none of
/// the runtime's locks, in submission order; each one's transition to `Scheduled` goes on after
/// its own (`CommandBufferShared::run_submission_handlers`). Each hold goes after its handlers,
/// outside every lock: it may be its command buffer's last reference.
fn run_submission_handlers(handlers: Handlers) {
    for buffer in handlers {
        buffer.run_submission_handlers();
    }
}

/// Batches for the thread to work on without its lock: each command buffer with the batch's
/// index.
type Steps = Vec<(Hold<CommandBufferShared>, usize)>;

/// Look at each line's head of a command buffer with a submission check that is otherwise
/// ready (its binary semaphores' signals submitted, its events' values reached, its host step
/// recorded, not `behind` an earlier commit) and waits for its checks: once every value it waits
/// for was reached on the device or by the host, not only by a submitted GPU signal (or the device
/// is lost), it goes to `checks`, unless there already. `gated` is set before the events are read
/// again, as [`Scheduler::completions`] says. Whether a head still waits for a GPU signal to land.
fn check_heads(state: &mut State, behind: &[u64], checks: &mut Steps, gated: &AtomicBool) -> bool {
    let State { order, binary, .. } = state;
    let mut landing = false;
    for mut line in order.lines_mut(behind) {
        let Some(held) = line.next() else {
            continue;
        };
        // Whether the head waits for nothing but its checks, if any: its followers' may then
        // run in this pass, after its own (`check_followers`).
        let clear = binary.ready(&held.binary_needs)
            && !(held.segments && held.buffer.needs_host_step(held.index));
        let due = held.checked
            && clear
            && held.buffer.needs_check(held.index)
            && !checks
                .iter()
                .any(|(buffer, index)| buffer.same(&held.buffer) && *index == held.index);
        let Some(waits) = event_waits(held) else {
            continue;
        };
        let mut waiting = !waits.is_empty();
        if due {
            // As before for a head: one whose waits only a GPU signal not yet landed meets
            // waits at its gate, unless the device is lost.
            if waiting && !held.buffer.device().is_lost() {
                gated.store(true, Ordering::SeqCst);
                fence(Ordering::SeqCst);
                waiting = event_waits(held).is_some_and(|waits| !waits.is_empty());
                if waiting {
                    landing = true;
                    continue;
                }
            }
            checks.push((held.buffer.hold(), held.index));
        }
        if clear && !waiting {
            check_followers(&mut line, binary, checks);
        }
    }
    landing
}

/// How many batches after a line's head have their checks run in a pass with the head's.
const CHECKED_AHEAD: usize = 15;

/// The batches after a line's head (`line`) that wait for nothing but their checks: each one's
/// binary semaphores' signals submitted, its host step recorded and every event value it waits
/// for reached by the host or the device, up to the first that does not. Their checks go to
/// `checks` with the head's, in line order, so that the next pass finds them ready one after
/// another and submits them as one submission (`coalesce.rs`). A check runs before its batch is
/// submitted and after those of the batches before it, as for a head; only sooner.
fn check_followers<'a>(
    line: &mut impl Iterator<Item = &'a mut Held>,
    binary: &BinarySignals<vk::Semaphore>,
    checks: &mut Steps,
) {
    let mut followers = Vec::new();
    for held in line.take(CHECKED_AHEAD) {
        if !binary.ready(&held.binary_needs)
            || (held.segments && held.buffer.needs_host_step(held.index))
            || !event_waits(held).is_some_and(|waits| waits.is_empty())
        {
            break;
        }
        if held.checked
            && held.buffer.needs_check(held.index)
            && !checks
                .iter()
                .any(|(buffer, index)| buffer.same(&held.buffer) && *index == held.index)
        {
            followers.push((held.buffer.hold(), held.index));
        }
    }
    checks.extend(followers);
}

/// Look at each line's head that is otherwise ready (not `behind` an earlier commit either,
/// its gate then covers that commit's work) and needs a host step: its gate is taken (the queues'
/// submitted values, the first time), and once every `VkQueue`'s completed value reached it (or the
/// device is lost), its step goes to `steps`, unless there already. `gated` is set before the
/// completed values are read, as [`Scheduler::completions`] says. Whether a head still waits at its
/// gate.
fn gate_heads(state: &mut State, behind: &[u64], steps: &mut Steps, gated: &AtomicBool) -> bool {
    let State {
        order,
        values,
        binary,
        ..
    } = state;
    let mut waiting = false;
    for held in order.heads_mut(behind) {
        if !held.segments
            || !binary.ready(&held.binary_needs)
            || event_waits(held).is_none()
            || !held.buffer.needs_host_step(held.index)
            || steps
                .iter()
                .any(|(buffer, index)| buffer.same(&held.buffer) && *index == held.index)
        {
            continue;
        }
        // A step waits for the pipelines its segments record, whose makings wake the thread as
        // they finish; one for them alone waits for no work on the GPU.
        let waits = held.buffer.step_waits(held.index);
        if waits.pipelines {
            continue;
        }
        if !waits.gpu {
            steps.push((held.buffer.hold(), held.index));
            continue;
        }
        let gate = held.gate.get_or_insert_with(|| values.clone());
        gated.store(true, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        let device = held.buffer.device();
        let shared = device.shared();
        let reached = device.is_lost()
            || shared
                .queues
                .iter()
                .zip(gate.iter())
                .all(|(queue, &value)| queue.completed_value() >= value);
        if reached {
            steps.push((held.buffer.hold(), held.index));
        } else {
            waiting = true;
        }
    }
    waiting
}

/// Whether the head of commit sequence number `sequence` waits behind a batch committed before it
/// that keeps its place ([`Order::is_behind`]): one whose binary semaphores' signals are submitted
/// and whose event values are reached, as is every batch before it on its line, and whose footprint
/// and the head's run in commit order ([`tracker::ordered`]). `memo` holds what this pass found.
fn is_behind(
    order: &Order<Held>,
    binary: &BinarySignals<vk::Semaphore>,
    memo: &mut Behind<Option<Footprint<ResourceKey>>>,
    sequence: u64,
) -> bool {
    order.is_behind(
        sequence,
        memo,
        (
            &mut |held: &Held| binary.ready(&held.binary_needs) && events_reachable(held),
            &mut |held: &Held| held.buffer.footprint(held.index),
            &|earlier: &Option<Footprint<ResourceKey>>, later: &Option<Footprint<ResourceKey>>| {
                tracker::ordered(earlier.as_ref(), later.as_ref())
            },
        ),
    )
}

/// Whether every event value `held` waits for is reached by a submitted signal or the host:
/// [`event_waits`] would answer, for a batch the order does not take yet.
fn events_reachable(held: &Held) -> bool {
    held.waits
        .iter()
        .all(|(event, value)| event.wait(*value).is_some())
}

/// The Vulkan waits of `held` for its events' signal indices, when every value it waits for is
/// reached by a submitted signal or the host; `None` while one is not.
fn event_waits(held: &Held) -> Option<Vec<(vk::Semaphore, u64)>> {
    let mut waits = Vec::new();
    for (event, value) in &held.waits {
        match event.wait(*value)? {
            WaitFor::Reached => {}
            WaitFor::Index(index) => waits.push((event.semaphore(), index)),
        }
    }
    Some(waits)
}
