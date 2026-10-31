//! What a render pipeline state's creation works out from its descriptor before its compile,
//! kept by the descriptor's key, in the device's memory and in the cache directory.
//!
//! A creation decoded and specialized its stages' AIR to answer Metal's linked-use refusals
//! (`RenderPipelineDescriptor::{vertex_returns_value, fragment_writes_depth_unattached,
//! stages_using_direct_arguments, unwritten_fragment_inputs}`), to learn the key its pipeline
//! is found by (`Device::render_variant_key`), and, where asked, its reflection: 18 to 23 ms of
//! a creation of the consumer's heaviest states on the RTX 5080 Laptop GPU's host made again in
//! the same process, and most of the 33 to 39 ms of one in a new process over warm caches (EV
//! SWA-wm1), where Metal answers in under 1 ms.
//!
//! - **Keyed by everything the answers read.** Each answer is a function of the descriptor and
//!   the device alone: `Device::render_descriptor_key` digests every field of the descriptor
//!   (each `Function` stage by its compile key: AIR hash, entry, constants, linked functions and
//!   dynamic libraries) with the device's profile fingerprint and `SEMANTICS_VERSION`. A
//!   pipeline key is kept per variant key too. The creation's budget decides only whether an
//!   answer comes at all: a refused answer (a deadline, a cancellation, a decode's error) is not
//!   kept, so the next creation asks again.
//! - **In memory first, then the directory.** A lookup asks the device's maps, then the
//!   directory's entry (kept in the maps when found); what a creation works out goes to both.
//!   The entries live in the build's directory like the artifacts, so a build never reads
//!   another's, under names of their kind, key and profile. The answers and the reflection are
//!   the translator's frames (`LinkedUseAnswers::to_bytes`,
//!   `RenderPipelineReflection::to_bytes`: versions, kind, key, profile and checksum checked); a
//!   variant's pipeline key is a record of its own name and the key under a SHA-256 checksum. A
//!   lookup counts in no statistic but a refused or unreadable entry, and a write in none but a
//!   failed one: they are not translations found, compiled or written.
//! - **Bounded.** Each map keeps at most [`ENTRIES`] entries, the least recently used going first
//!   ([`Lru`]); the directory's entries go as every entry does, past its cap. They hold digests,
//!   small answers and reflections, and no AIR, library, or Vulkan object.

use super::store::{Lookup, entry_name};
use super::{EntryKind, ShaderCache};
use crate::device::lock;
use crate::pipeline::memo::Lru;
use metal2vulkan::cache::{Hash256, PipelineKey, VariantKey};
use metal2vulkan::device::Device as CompileDevice;
use metal2vulkan::library::FunctionCompileKey;
use metal2vulkan::reflection::{LinkedUseAnswers, RenderPipelineReflection};
use sha2::{Digest, Sha256};
use std::sync::Mutex;

/// The most entries each of the memo's maps keeps.
pub(crate) const ENTRIES: usize = 256;

/// The tag a variant's record name hashes first.
const VARIANT_NAME_TAG: &[u8] = b"metal2vulkan-runtime variant pipeline name v2\0";

/// The first bytes of a variant's record.
const VARIANT_MAGIC: &[u8; 8] = b"M2VVPIP\x02";

/// A variant record's length: its magic, its name, the pipeline key, and the checksum.
const VARIANT_RECORD: usize = 8 + 32 + 32 + 32 + 1 + 32 + 32;

/// Stage compile keys cached alongside the fixed-function pipeline key.
/// Persisting them permits re-linking without decoding Metal AIR a second time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct VariantStages {
    pub(crate) pipeline: PipelineKey,
    pub(crate) vertex: FunctionCompileKey,
    pub(crate) fragment: Option<FunctionCompileKey>,
}

/// The device's memo of what its render creations worked out before their compiles (the module
/// docs).
pub(crate) struct CreationMemo {
    linked: Mutex<Lru<PipelineKey, LinkedUseAnswers>>,
    pipelines: Mutex<Lru<(PipelineKey, VariantKey), VariantStages>>,
    reflections: Mutex<Lru<PipelineKey, RenderPipelineReflection>>,
}

impl CreationMemo {
    /// An empty memo.
    pub(crate) fn new() -> CreationMemo {
        CreationMemo {
            linked: Mutex::new(Lru::counted(ENTRIES)),
            pipelines: Mutex::new(Lru::counted(ENTRIES)),
            reflections: Mutex::new(Lru::counted(ENTRIES)),
        }
    }
}

impl ShaderCache {
    /// The linked-use answers a creation of the render descriptor of `key` on `compile` kept:
    /// the memo's, else the directory's.
    pub(crate) fn linked_answers(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
    ) -> Option<LinkedUseAnswers> {
        if let Some(answers) = lock(&self.creation.linked).get(key) {
            return Some(answers.clone());
        }
        let answers =
            self.read_answer(EntryKind::LinkedAnswers, key.digest(), compile, |bytes| {
                LinkedUseAnswers::from_bytes_for_key(bytes, key, compile)
            })?;
        lock(&self.creation.linked).insert(*key, answers.clone(), 1);
        Some(answers)
    }

    /// Keep `answers` as the linked-use answers of the render descriptor of `key` on `compile`,
    /// in the memo and the directory.
    pub(crate) fn keep_linked_answers(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        answers: LinkedUseAnswers,
    ) {
        let bytes = answers.to_bytes(key, compile);
        lock(&self.creation.linked).insert(*key, answers, 1);
        self.keep_answer(EntryKind::LinkedAnswers, key.digest(), compile, &bytes);
    }

    /// Variant keys in memory, else in the directory under a versioned record.
    pub(crate) fn variant_pipeline(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        variant: &VariantKey,
    ) -> Option<VariantStages> {
        let at = (*key, variant.clone());
        if let Some(stages) = lock(&self.creation.pipelines).get(&at) {
            return Some(*stages);
        }
        let name = variant_name(key, variant);
        let stages = self.read_answer(EntryKind::VariantPipeline, &name, compile, |bytes| {
            variant_record_stages(bytes, &name)
        })?;
        lock(&self.creation.pipelines).insert(at, stages, 1);
        Some(stages)
    }

    /// Persist shader keys even when the pipeline artifact is not stored,
    /// so a warm process can find its functions and link without AIR decode.
    pub(crate) fn keep_variant_pipeline(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        variant: &VariantKey,
        stages: &VariantStages,
    ) {
        let at = (*key, variant.clone());
        {
            let mut pipelines = lock(&self.creation.pipelines);
            if pipelines.get(&at) == Some(stages) {
                return;
            }
            pipelines.insert(at, *stages, 1);
        }
        let name = variant_name(key, variant);
        self.keep_answer(
            EntryKind::VariantPipeline,
            &name,
            compile,
            &variant_record(&name, stages),
        );
    }

    /// The reflection of the render descriptor of `key` on `compile`: the memo's, else the
    /// directory's, else `work`'s, kept in both where it answers.
    pub(crate) fn render_reflection<E>(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        work: impl FnOnce() -> Result<RenderPipelineReflection, E>,
    ) -> Result<RenderPipelineReflection, E> {
        if let Some(reflection) = self.kept_render_reflection(compile, key) {
            return Ok(reflection);
        }
        Ok(self.keep_render_reflection(compile, key, work()?))
    }

    /// The reflection of the render descriptor of `key` on `compile` the memo keeps, else the
    /// directory's, then kept in the memo: [`render_reflection`](Self::render_reflection)'s
    /// answer where it needs no work.
    pub(crate) fn kept_render_reflection(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
    ) -> Option<RenderPipelineReflection> {
        if let Some(reflection) = lock(&self.creation.reflections).get(key) {
            return Some(reflection.clone());
        }
        let reflection = self.read_answer(
            EntryKind::RenderReflection,
            key.digest(),
            compile,
            |bytes| RenderPipelineReflection::from_bytes_for_key(bytes, key, compile),
        )?;
        lock(&self.creation.reflections).insert(*key, reflection.clone(), 1);
        Some(reflection)
    }

    /// Keep `reflection`, worked out for the render descriptor of `key` on `compile`, in the memo
    /// and the directory; returned.
    pub(crate) fn keep_render_reflection(
        &self,
        compile: &CompileDevice,
        key: &PipelineKey,
        reflection: RenderPipelineReflection,
    ) -> RenderPipelineReflection {
        let bytes = reflection.to_bytes(key, compile);
        lock(&self.creation.reflections).insert(*key, reflection.clone(), 1);
        self.keep_answer(EntryKind::RenderReflection, key.digest(), compile, &bytes);
        reflection
    }

    /// Write `bytes` as the directory's entry of `kind` under `key` for `compile`'s profile, as
    /// an artifact is written (the bytes counted against the directory's cap, the least recently
    /// used entries going past it), but counted as no entry written (it is no translation); a
    /// failed write is counted and said as an artifact's is.
    fn keep_answer(&self, kind: EntryKind, key: &Hash256, compile: &CompileDevice, bytes: &[u8]) {
        let name = entry_name(
            self.store.directory().build(),
            kind,
            key,
            &compile.profile_fingerprint(),
        );
        if self.store.write(&name, |out| out.write_all(bytes)).is_err() {
            self.store.write_failed();
        }
        if self.store.over_capacity() {
            self.worker.evict();
        }
    }

    /// The directory's entry of `kind` under `key` for `compile`'s profile, as `decode` accepts
    /// it: counted as a hit or a miss by no statistic (it is no translation), a refused or
    /// unreadable entry counted as the store counts it.
    fn read_answer<T, E>(
        &self,
        kind: EntryKind,
        key: &Hash256,
        compile: &CompileDevice,
        decode: impl FnOnce(&[u8]) -> Result<T, E>,
    ) -> Option<T> {
        let name = entry_name(
            self.store.directory().build(),
            kind,
            key,
            &compile.profile_fingerprint(),
        );
        match self.store.load(&name, decode) {
            Lookup::Hit(value) => Some(value),
            Lookup::Absent | Lookup::Unreadable | Lookup::Rejected => None,
        }
    }
}

/// The name a variant's record is kept under: the SHA-256 of the render descriptor's key and the
/// variant key's every field (its derived `Debug` rendering, as the translator's keys render
/// their inputs).
fn variant_name(key: &PipelineKey, variant: &VariantKey) -> Hash256 {
    let mut hash = Sha256::new();
    hash.update(VARIANT_NAME_TAG);
    hash.update(key.digest());
    hash.update(format!("{variant:?}").as_bytes());
    hash.finalize().into()
}

/// Record pipeline and shader-stage keys under a checksummed v2 envelope.
fn variant_record(name: &Hash256, stages: &VariantStages) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(VARIANT_RECORD);
    bytes.extend_from_slice(VARIANT_MAGIC);
    bytes.extend_from_slice(name);
    bytes.extend_from_slice(stages.pipeline.digest());
    bytes.extend_from_slice(stages.vertex.digest());
    bytes.push(u8::from(stages.fragment.is_some()));
    bytes.extend_from_slice(stages.fragment.as_ref().map_or(&[0; 32], |f| f.digest()));
    let checksum: Hash256 = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&checksum);
    bytes
}

/// A corrupt, wrong-name, or old-version record cannot be used.
#[derive(Debug)]
struct NotTheRecord;

fn variant_record_stages(bytes: &[u8], name: &Hash256) -> Result<VariantStages, NotTheRecord> {
    let sum = VARIANT_RECORD - 32;
    if bytes.len() != VARIANT_RECORD
        || &bytes[..8] != VARIANT_MAGIC
        || &bytes[8..40] != name
        || bytes[sum..] != <[u8; 32]>::from(Sha256::digest(&bytes[..sum]))
    {
        return Err(NotTheRecord);
    }
    let digest = |at: usize| -> Result<Hash256, NotTheRecord> {
        bytes[at..at + 32].try_into().map_err(|_| NotTheRecord)
    };
    let fragment = match bytes[104] {
        0 => None,
        1 => Some(FunctionCompileKey::new(digest(105)?)),
        _ => return Err(NotTheRecord),
    };
    Ok(VariantStages {
        pipeline: PipelineKey::new(digest(40)?),
        vertex: FunctionCompileKey::new(digest(72)?),
        fragment,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn variant_records_roundtrip_and_reject_corruption() {
        let key = PipelineKey::new([1; 32]);
        let mut variant = VariantKey::new();
        let name = variant_name(&key, &variant);
        variant.emulated_simdgroups = true;
        let other = variant_name(&key, &variant);
        assert_ne!(name, other);
        for fragment in [None, Some(FunctionCompileKey::new([4; 32]))] {
            let stages = VariantStages {
                pipeline: PipelineKey::new([9; 32]),
                vertex: FunctionCompileKey::new([3; 32]),
                fragment,
            };
            let record = variant_record(&name, &stages);
            assert_eq!(variant_record_stages(&record, &name).ok(), Some(stages));
            assert!(variant_record_stages(&record, &other).is_err());
            for at in [0, 8, 40, 72, 104, 105, VARIANT_RECORD - 1] {
                let mut damaged = record.clone();
                damaged[at] ^= 1;
                assert!(variant_record_stages(&damaged, &name).is_err());
            }
            assert!(variant_record_stages(&record[..VARIANT_RECORD - 1], &name).is_err());
            let mut bad_flag = record.clone();
            bad_flag[104] = 2;
            let sum = VARIANT_RECORD - 32;
            let checksum: Hash256 = Sha256::digest(&bad_flag[..sum]).into();
            bad_flag[sum..].copy_from_slice(&checksum);
            assert!(variant_record_stages(&bad_flag, &name).is_err());
        }
    }
}
