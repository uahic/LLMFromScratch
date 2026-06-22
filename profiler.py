import cProfile
import pstats
import io
from functools import wraps


def profile(func=None, output_file: str | None = None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = f(*args, **kwargs)

            profiler.disable()

            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
            ps.print_stats(40)
            print(s.getvalue())

            if output_file:
                ps.dump_stats(output_file)
                print(f"Profile has been saved to {output_file}")
            return result

        return wrapper

    if func is None:
        return decorator
    return decorator(func)
