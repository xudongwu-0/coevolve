from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sao_bridge.analysis import expected_sample_ledger
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.paired_campaign import trial_spec, verify_peer, verify_unique_recycling


class PairedCampaignTests(unittest.TestCase):
    def protocol(self):
        return read_json(Path(__file__).resolve().parents[1] / "examples/sft1152_hwh_pair_20260929.json")

    def test_concurrent_same_sft_seed_budget_and_no_formal_reference_dependency(self):
        protocol = self.protocol()
        receipt = {"status": "completed", "hf_cpu_reload_verified": True,
                   "checkpoint": protocol["model"], "cumulative_train_records": 1152}
        with patch("sao_bridge.paired_campaign.read_json", return_value=receipt):
            a, b = (trial_spec(protocol, host) for host in ("dev-NDH200-026", "dev-NDH200-025"))
        self.assertEqual((a["method"], b["method"]), ("independent", "ma_sao"))
        self.assertEqual((a["seed"], b["seed"]), (42, 42))
        self.assertEqual(a["model"], b["model"])
        self.assertNotIn("reference_run", b)
        left, right = expected_sample_ledger(a), expected_sample_ledger(b)
        self.assertEqual(left["generated_total"], 31440)
        self.assertEqual(left["generated_total"] - right["generated_total"], 80)
        self.assertEqual(right["recycled_partner_episodes_reused"], 3072)
        self.assertFalse(a["evaluate_test"] or b["formal_baseline"])

    def test_original_model_is_not_silently_substituted_for_sft(self):
        protocol = self.protocol()
        receipt = {"status": "completed", "hf_cpu_reload_verified": True,
                   "checkpoint": "original", "cumulative_train_records": 1152}
        with patch("sao_bridge.paired_campaign.read_json", return_value=receipt):
            with self.assertRaisesRegex(ValueError, "SFT"):
                trial_spec(protocol, "dev-NDH200-026")
        receipt["checkpoint"] = protocol["model"]
        protocol["common"]["reward_questions"] = 16
        with patch("sao_bridge.paired_campaign.read_json", return_value=receipt):
            with self.assertRaisesRegex(ValueError, "partner pool"):
                trial_spec(protocol, "dev-NDH200-025")

    def test_peer_must_match_seed_identity_and_unchanged_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {"protocol_sha256": "p", "initial_actor_sha256": "m", "data_manifest_sha256": "d"}
            write_json(root / "COMPLETE.json", {"method": "independent"})
            record = {**identity, "method": "independent", "seed": 42, "run": str(root),
                      "reports": {"COMPLETE.json": file_sha(root / "COMPLETE.json")}}
            marker = root / "READY.json"
            write_json(marker, record)
            verify_peer(marker, identity, 42)
            with self.assertRaisesRegex(ValueError, "seed"):
                verify_peer(marker, identity, 43)
            write_json(root / "COMPLETE.json", {"method": "altered"})
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_peer(marker, identity, 42)

    def test_actual_recycling_rejects_repeated_or_cross_pool_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = dict(critic_warmup_steps=1, solver_batch_size=2,
                        warmup_recycled_per_row=1, solver_steps=1, solver_recycle_fraction=1.0)
            rows = [{"warmup": True, "recycled_ids": ["w0", "w1"]},
                    {"warmup": False, "recycled_ids": ["m0", "m1"]}]
            write_json(root / "w1_solver_data.json", rows)
            write_json(root / "w1_recycle.json", [{"id": x} for x in ("w0", "w1", "m0", "m1")])
            write_json(root / "w1/trained.json", {"actor_steps": 1, "critic_only_warmup_steps": 1})
            self.assertEqual(verify_unique_recycling(root, spec)["max_reuse"], 1)
            rows[1]["recycled_ids"] = ["w0", "m1"]
            write_json(root / "w1_solver_data.json", rows)
            with self.assertRaisesRegex(ValueError, "once-per-episode"):
                verify_unique_recycling(root, spec)


if __name__ == "__main__":
    unittest.main()
