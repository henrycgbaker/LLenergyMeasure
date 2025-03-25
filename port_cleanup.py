import subprocess
import os
import signal
import time

def cleanup_port(port, wait_seconds=0.5):
    try:
        # Get the list of PIDs using the port.
        result = subprocess.check_output(["lsof", "-ti", f":{port}"])
        # Filter out any empty strings and convert to int.
        pids = [int(pid) for pid in result.decode("utf-8").strip().split("\n") if pid]
        if not pids:
            print(f"No processes found on port {port}.")
            return

        print(f"Found processes on port {port}: {pids}")

        # Send SIGTERM to each process.
        for pid in pids:
            try:
                details = subprocess.check_output(["ps", "-p", str(pid), "-o", "comm="])
                process_name = details.decode("utf-8").strip()
                print(f"Process {pid} running: {process_name}")
            except subprocess.CalledProcessError as detail_error:
                print(f"Could not get details for PID {pid}: {detail_error}")

            try:
                print(f"Sending SIGTERM to PID {pid}")
                os.kill(pid, signal.SIGTERM)
            except Exception as kill_error:
                print(f"Error sending SIGTERM to PID {pid}: {kill_error}")

        # Wait for a bit to allow processes to terminate gracefully.
        time.sleep(wait_seconds)

        # Check each PID and if still alive, send SIGKILL.
        for pid in pids:
            try:
                # os.kill with signal 0 will raise an OSError if the process is gone.
                os.kill(pid, 0)
            except OSError:
                print(f"PID {pid} terminated successfully after SIGTERM.")
            else:
                try:
                    print(f"PID {pid} still running; sending SIGKILL")
                    os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    print(f"Failed to kill PID {pid} with SIGKILL: {e}")

        print("Cleanup complete.")
    except subprocess.CalledProcessError as e:
        print(f"No processes found on port {port} or error occurred: {e}")

cleanup_port(29500)
