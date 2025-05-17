#!/usr/bin/env python3
import subprocess
import os
import signal
import time
import pwd

def get_username_from_pid(pid: int) -> str:
    """
    Returns the username owning the given pid.
    """
    try:
        out = subprocess.check_output(
            ["ps", "-o", "user=", "-p", str(pid)],
            text=True
        ).strip()
        return out
    except subprocess.CalledProcessError:
        return ""

def cleanup_port(port: int, wait_seconds: float = 0.5, kill_parents: bool = True):
    """
    Finds processes listening on `port`, logs their names and parents,
    then sends SIGTERM → SIGKILL to both child and (optionally) parent
    if they’re owned by the current user.
    """
    user = pwd.getpwuid(os.getuid()).pw_name

    try:
        # 1) find all PIDs on the port
        raw = subprocess.check_output(["lsof", "-ti", f":{port}"], text=True)
        pids = [int(x) for x in raw.strip().splitlines() if x.strip()]
        if not pids:
            print(f"No processes found on port {port}.")
            return

        print(f"Found processes on port {port}: {pids}")

        to_kill = set(pids)  # child PIDs
        parent_map = {}

        # 2) for each pid, find its command, its ppid and parent command
        for pid in pids:
            try:
                # get command name
                cmd = subprocess.check_output(
                    ["ps", "-p", str(pid), "-o", "comm="],
                    text=True
                ).strip()

                # get PPID
                ppid = int(subprocess.check_output(
                    ["ps", "-p", str(pid), "-o", "ppid="],
                    text=True
                ).strip())

                # get parent command
                parent_cmd = subprocess.check_output(
                    ["ps", "-p", str(ppid), "-o", "comm="],
                    text=True
                ).strip()

                print(f"→ PID {pid} ({cmd}), parent PID {ppid} ({parent_cmd})")

                parent_map[pid] = (ppid, parent_cmd)

                # if configured, also kill the parent if same user
                if kill_parents:
                    parent_user = get_username_from_pid(ppid)
                    if parent_user == user:
                        to_kill.add(ppid)
                    else:
                        print(f"   └─ skipping parent {ppid} (owned by {parent_user})")

            except subprocess.CalledProcessError:
                print(f"Could not inspect PID {pid}. It may have exited already.")

        # 3) send SIGTERM
        for pid in list(to_kill):
            try:
                print(f"Sending SIGTERM to PID {pid}")
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                print(f"PID {pid} not found (already exited).")
            except PermissionError:
                print(f"Permission denied sending SIGTERM to PID {pid}.")

        # give them a moment to shut down gracefully
        time.sleep(wait_seconds)

        # 4) send SIGKILL to anything still alive
        for pid in list(to_kill):
            try:
                os.kill(pid, 0)  # check existence
            except OSError:
                print(f"PID {pid} terminated successfully after SIGTERM.")
            else:
                try:
                    print(f"PID {pid} still alive; sending SIGKILL")
                    os.kill(pid, signal.SIGKILL)
                except PermissionError:
                    print(f"Permission denied sending SIGKILL to PID {pid}.")
                except ProcessLookupError:
                    print(f"PID {pid} already gone.")

        print("Cleanup complete.")

    except subprocess.CalledProcessError as e:
        print(f"No processes found on port {port} or error occurred: {e}")

if __name__ == "__main__":
    cleanup_port(29500)
