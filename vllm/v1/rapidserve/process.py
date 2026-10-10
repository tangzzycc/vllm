# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import traceback
from multiprocessing.connection import Connection

from vllm.config import set_current_vllm_config
from vllm.distributed import destroy_distributed_environment, destroy_model_parallel
from vllm.v1.executor.uniproc_executor import UniProcExecutor
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
            while True:
                method, args = connection.recv()
                if method == "close":
                    break
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
