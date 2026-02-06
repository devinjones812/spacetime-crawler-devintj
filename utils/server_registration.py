import os
import time
from spacetime import Node
from utils.pcc_models import Register

def init(df, user_agent, fresh):
    reg = df.read_one(Register, user_agent)
    if not reg:
        reg = Register(user_agent, fresh)
        df.add_one(Register, reg)
        df.commit()
        df.push_await()

    # Timeout after 30 seconds of waiting
    timeout = 30
    start_time = time.time()
    attempts = 0

    while not reg.load_balancer:
        attempts += 1
        elapsed = time.time() - start_time

        if elapsed > timeout:
            raise RuntimeError(
                f"\n{'='*80}\n"
                f"ERROR: Cache server connection timeout after {timeout} seconds.\n"
                f"Made {attempts} attempts to connect to the server.\n\n"
                f"The UCI cache server (styx.ics.uci.edu:9000) is likely DOWN.\n\n"
                f"What to do:\n"
                f"  1. Check Ed Discussion for server status updates\n"
                f"  2. Ask your TA/instructor if the server is running\n"
                f"  3. Verify your user agent in config.ini is correct\n"
                f"  4. Try again later when the server is back online\n"
                f"{'='*80}\n"
            )

        df.pull_await()
        if reg.invalid:
            raise RuntimeError("User agent string is not acceptable.")
        if reg.load_balancer:
            df.delete_one(Register, reg)
            df.commit()
            df.push()
    return reg.load_balancer

def get_cache_server(config, restart):
    init_node = Node(
        init, Types=[Register], dataframe=(config.host, config.port))
    return init_node.start(
        config.user_agent, restart or not os.path.exists(config.save_file))