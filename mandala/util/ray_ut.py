from ..common_imports import *
from ..core.config import EnvConfig
if EnvConfig.has_ray:
    import ray

    def track_ray_progress(futures:TList[ray.ObjectRef]) -> list:
        done = False
        num_futures = len(futures)
        while not done:
            # require all results with a small timeout
            finished, pending = ray.wait(futures,
                                         num_returns=num_futures, timeout=0.1)
            time.sleep(5)
            if not pending:
                done = True
            print(f'Finished: {len(finished)}, pending: {len(pending)}')
        results = ray.get(futures)
        return results