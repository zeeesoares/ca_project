"""
Integration tests for orchestrator-migrater communication.

Each test starts the relevant server in a background thread, exercises the
client, and then shuts everything down cleanly.
"""

import time
import unittest
from concurrent import futures

import grpc

from protocol import cluster_pb2, cluster_pb2_grpc
from protocol.migrater import migrater_pb2, migrater_pb2_grpc
from protocol.orchestrator.client import OrchestratorClient
from protocol.migrater.client import MigraterClient
from orchestrator.scheduler import (
    ClusterState,
    WorkerState,
    UniformBandwidthPolicy,
)


# Helpers

def _start_server(server, port):
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    time.sleep(0.05)
    return server


def _make_workers(**kwargs) -> dict:
    """Build a workers snapshot dict. Each kwarg: worker_id=(pending_bytes, is_migrating)."""
    return {
        wid: WorkerState(worker_id=wid, pending_data_size=p, is_migrating=m)
        for wid, (p, m) in kwargs.items()
    }


MB = 1024 * 1024
GB = 1024 * MB


# Migrater integration tests

class TestMigraterCommunication(unittest.TestCase):

    PORT = 15051

    def setUp(self):
        from migrater.server import MigraterService as _MigraterService
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        migrater_pb2_grpc.add_MigraterServiceServicer_to_server(_MigraterService(), self.server)
        _start_server(self.server, self.PORT)
        self.client = MigraterClient(addr=f"localhost:{self.PORT}")

    def tearDown(self):
        self.server.stop(grace=1)

    def test_notify_returns_ok(self):
        request = migrater_pb2.CheckpointSavedRequest(
            checkpoint_local_path="/tmp/ckpt.pt",
            checkpoint_pfs_path="/pfs/ckpt.pt",
            timestamp=time.time(),
        )
        stub = migrater_pb2_grpc.MigraterServiceStub(
            grpc.insecure_channel(f"localhost:{self.PORT}")
        )
        response = stub.NotifyCheckpointSaved(request)
        self.assertTrue(response.ok)

    def test_notify_via_client_no_error(self):
        self.client.notify_checkpoint_saved("/tmp/ckpt2.pt", "/pfs/ckpt2.pt", time.time())


# ---------------------------------------------------------------------------
# Orchestrator integration tests (using UniformBandwidthPolicy, the default)
# ---------------------------------------------------------------------------

class TestOrchestratorCommunication(unittest.TestCase):

    PORT = 15052

    def setUp(self):
        from orchestrator.server import OrchestratorService
        self.cluster = ClusterState()
        self.server  = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        cluster_pb2_grpc.add_OrchestratorServiceServicer_to_server(
            OrchestratorService(UniformBandwidthPolicy(pfs_bandwidth_bps=1 * 1024 * 1024 * 1024), self.cluster),
            self.server,
        )
        _start_server(self.server, self.PORT)
        self.client = OrchestratorClient(worker_id="test-worker-0", addr=f"localhost:{self.PORT}")

    def tearDown(self):
        self.client.close()
        self.server.stop(grace=1)

    def test_single_heartbeat_returns_instruction(self):
        instructions = list(self.client.monitor([(0.0, False)]))
        self.assertEqual(len(instructions), 1)

    def test_hold_when_no_pending_data(self):
        instructions = list(self.client.monitor([(0.0, False)]))
        self.assertEqual(instructions[0].action, cluster_pb2.InstructionResponse.HOLD)

    def test_start_flush_when_above_threshold(self):
        instructions = list(self.client.monitor([(200 * MB, False)]))
        self.assertEqual(instructions[0].action, cluster_pb2.InstructionResponse.START_FLUSH)

    def test_multiple_heartbeats(self):
        heartbeats = [(0.0, False), (50 * MB, False), (200 * MB, False)]
        instructions = list(self.client.monitor(heartbeats))
        self.assertEqual(len(instructions), 3)
        self.assertEqual(instructions[0].action, cluster_pb2.InstructionResponse.HOLD)
        self.assertEqual(instructions[1].action, cluster_pb2.InstructionResponse.HOLD)
        self.assertEqual(instructions[2].action, cluster_pb2.InstructionResponse.START_FLUSH)

    def test_worker_removed_on_disconnect(self):
        list(self.client.monitor([(0.0, False)]))
        time.sleep(0.1)
        self.assertNotIn("test-worker-0", self.cluster.snapshot())


# ---------------------------------------------------------------------------
# Policy unit tests (no gRPC — just call decide() directly)
# ---------------------------------------------------------------------------

class TestUniformBandwidthPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = UniformBandwidthPolicy(pfs_bandwidth_bps=1 * GB, threshold_bytes=100 * MB)

    def test_single_worker_gets_full_bandwidth(self):
        workers = _make_workers(w0=(200 * MB, False))
        r = self.policy.decide("w0", workers)
        self.assertEqual(r.action, cluster_pb2.InstructionResponse.START_FLUSH)
        self.assertAlmostEqual(r.rate_limit_bps, 1 * GB)

    def test_two_workers_split_bandwidth_equally(self):
        workers = _make_workers(w0=(200 * MB, False), w1=(200 * MB, False))
        r0 = self.policy.decide("w0", workers)
        r1 = self.policy.decide("w1", workers)
        self.assertAlmostEqual(r0.rate_limit_bps, 0.5 * GB)
        self.assertAlmostEqual(r1.rate_limit_bps, 0.5 * GB)

    def test_hold_for_worker_below_threshold(self):
        workers = _make_workers(w0=(200 * MB, False), w1=(10 * MB, False))
        r1 = self.policy.decide("w1", workers)
        self.assertEqual(r1.action, cluster_pb2.InstructionResponse.HOLD)

    def test_migrating_worker_gets_change_rate(self):
        workers = _make_workers(w0=(200 * MB, True))
        r = self.policy.decide("w0", workers)
        self.assertEqual(r.action, cluster_pb2.InstructionResponse.CHANGE_RATE)


# ---------------------------------------------------------------------------
# ClusterState unit tests
# ---------------------------------------------------------------------------

class TestClusterState(unittest.TestCase):

    def test_update_and_snapshot(self):
        cs = ClusterState()
        cs.update("w0", 100.0, False)
        snap = cs.snapshot()
        self.assertIn("w0", snap)
        self.assertEqual(snap["w0"].pending_data_size, 100.0)

    def test_remove(self):
        cs = ClusterState()
        cs.update("w0", 100.0, False)
        cs.remove("w0")
        self.assertNotIn("w0", cs.snapshot())

    def test_snapshot_is_copy(self):
        cs = ClusterState()
        cs.update("w0", 100.0, False)
        snap = cs.snapshot()
        cs.update("w0", 999.0, True)
        self.assertEqual(snap["w0"].pending_data_size, 100.0)  # snapshot is frozen


if __name__ == "__main__":
    unittest.main(verbosity=2)
