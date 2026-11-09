"""Unit tests for the read-only RADV scheduler stats comparator."""
import unittest

from amd_rdna_compare import comparison, delta_percent, parse_line, summarize


def line(submits: int, batches: int, mean_us: int, limit: int = 0) -> str:
    return (
        f"qemu: m2v-sched: {submits} submits, {batches} batches "
        f"({batches / submits:.1f} a submit), {mean_us} us a submit; "
        f"ends: queue 0 limit {limit} conflict 3 fence 4 pass 5"
    )


class SchedulerComparatorTests(unittest.TestCase):
    def test_extracts_stats_from_prefixed_log(self):
        record = parse_line(line(10, 70, 120, 2))
        self.assertIsNotNone(record)
        self.assertEqual(record["submits"], 10)
        self.assertEqual(record["batches"], 70)
        self.assertEqual(record["mean_us"], 120)
        self.assertEqual(record["limit"], 2)
        self.assertIsNone(parse_line("kernel: amdgpu: queue busy"))

    def test_aggregate_uses_submit_weighted_mean(self):
        report = summarize([line(10, 70, 100), line(30, 240, 300)])
        self.assertEqual(report["reported_windows"], 2)
        self.assertEqual(report["total_submits"], 40)
        self.assertEqual(report["total_batches"], 310)
        self.assertEqual(report["batches_per_submit"], 7.75)
        self.assertEqual(report["estimated_mean_submit_us"], 250.0)
        self.assertEqual(report["estimated_total_submit_time_ms"], 10.0)
        self.assertEqual(report["end_reasons"]["conflict"], 6)

    def test_refuses_empty_evidence(self):
        with self.assertRaises(ValueError):
            summarize(["no scheduler data"])
        with self.assertRaises(ValueError):
            summarize([line(1, 2, 100).replace("1 submits", "0 submits")])

    def test_percent_change_is_not_named_speedup(self):
        base = summarize([line(20, 40, 100)])
        exp = summarize([line(10, 50, 120)])
        changes = comparison(base, exp)
        self.assertEqual(changes["batches_per_submit"], 150.0)
        self.assertEqual(changes["estimated_mean_submit_us"], 20.0)
        self.assertEqual(changes["submits_per_reported_window"], -50.0)
        self.assertIsNone(delta_percent(0, 10))


if __name__ == "__main__":
    unittest.main()
