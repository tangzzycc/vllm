# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import traceback
from collections import deque
from concurrent.futures import Future
from contextlib import ExitStack, suppress
from multiprocessing.connection import Connection
from typing import Any

from vllm.config import set_current_vllm_config
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.v1.executor.multiproc_executor import MultiprocExecutor
from vllm.v1.executor.uniproc_executor import UniProcExecutor
from vllm.v1.rapidserve.parallel import ParallelWorker
from vllm.v1.rapidserve.prefill import PrefillEngine
from vllm.v1.rapidserve.worker import SharedModelWorker


def run_prefill(config, connection: Connection) -> None:
    """Own shared allocations until the decode process closes its mappings."""
    executor = None
    worker = None
    try:
        with set_current_vllm_config(config):
            if config.parallel_config.world_size > 1:
                executor = MultiprocExecutor(config)
                worker = ParallelWorker(executor)
            else:
                executor = UniProcExecutor(config)
                worker = executor.driver_worker.worker
                assert isinstance(worker, SharedModelWorker)
            connection.send(("weights", worker.export_weights()))
            engine = None
            quiesce_epoch = None
            policies: deque[tuple[int, Future[Any]]] = deque()
            while True:
                if policies and policies[0][1].done():
                    epoch, completion = policies.popleft()
                    completion.result()
                    connection.send(("policy_done", epoch))
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
                    result = worker.advance_resource_policy(*args)
                    if isinstance(result, Future):
                        policies.append((args[0], result))
                    else:
                        connection.send(("policy_done", result))
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
        with ExitStack() as cleanup:
            cleanup.callback(connection.close)
            cleanup.callback(destroy_distributed_environment)
            cleanup.callback(destroy_model_parallel)
            if isinstance(worker, ParallelWorker):
                cleanup.callback(worker.close)
            if executor is not None:
                executor.shutdown()
