# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import traceback
from contextlib import suppress
from multiprocessing.connection import Connection

from vllm.config import set_current_vllm_config
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.rapidserve.prefill import PrefillEngine
from vllm.v1.rapidserve.worker import SharedModelWorker


def run_prefill(config, connection: Connection) -> None:
    """Own shared allocations until the decode process closes its mappings."""
    executor = None
    try:
        with set_current_vllm_config(config):
            executor = UniProcExecutor(config)
            worker = executor.driver_worker.worker
            assert isinstance(worker, SharedModelWorker)
            connection.send(("weights", worker.export_weights()))
            engine = None
            quiesce_epoch = None
            while True:
                if (
                    quiesce_epoch is not None
                    and engine is not None
                    and not engine.pending
                ):
                    connection.send(("quiesced", quiesce_epoch))
                    quiesce_epoch = None
                if engine is not None and not connection.poll(
                    0 if engine.has_work() else 0.1
                ):
                    if not engine.step():
                        time.sleep(0.0001)
                    continue
                method, args = connection.recv()
                if method == "close":
                    break
                if method == "start":
                    engine = PrefillEngine(worker, connection)
                    connection.send(("result", None))
                    continue
                if method == "work":
                    assert engine is not None
                    engine.add(args[0])
                    engine.step()
                    continue
                if method == "cancel":
                    assert engine is not None
                    engine.cancel(args[0])
                    continue
                if method == "policy":
                    assert engine is not None and not engine.pending
                    assert engine.paused or not engine.has_work()
                    connection.send(("policy_done", worker.set_resource_policy(*args)))
                    continue
                if method == "advance_policy":
                    connection.send(
                        ("policy_done", worker.advance_resource_policy(*args))
                    )
                    continue
                if method == "quiesce":
                    assert engine is not None
                    engine.paused = True
                    quiesce_epoch = args[0]
                    continue
                if method == "resume":
                    assert engine is not None
                    engine.paused = False
                    continue
                result = getattr(worker, method)(*args)
                connection.send(("result", result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        with suppress(EOFError, BrokenPipeError, OSError):
            connection.send(("error", traceback.format_exc()))
            # Keep exported storage alive while decode tears down its mappings.
            deadline = time.monotonic() + 30
            while connection.poll(max(0, deadline - time.monotonic())):
                method, _ = connection.recv()
                if method == "close":
                    break
                if time.monotonic() >= deadline:
                    break
    finally:
        if executor is not None:
            executor.shutdown()
        destroy_model_parallel()
        destroy_distributed_environment()
        connection.close()
