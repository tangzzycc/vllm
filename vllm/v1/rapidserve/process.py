# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
import traceback
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
            while True:
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
                result = getattr(worker, method)(*args)
                connection.send(("result", result))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(("error", traceback.format_exc()))
    finally:
        if executor is not None:
            executor.shutdown()
            destroy_model_parallel()
            destroy_distributed_environment()
        connection.close()
