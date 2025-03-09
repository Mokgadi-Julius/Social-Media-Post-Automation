#!/usr/bin/env python3
"""
advanced_scheduler.py - Advanced scheduler for WriteNow Agency's social media content generators

This script:
1. Schedules Twitter content generation 4 times a day (SAST timezone)
2. Schedules Facebook and LinkedIn content generation 3 times a day
3. Provides status monitoring and reporting
4. Implements error handling with retry mechanisms
5. Maintains detailed logs with rotation
6. Can run as a background service with nohup
7. Includes health check and notification capabilities

Usage:
    python3 advanced_scheduler.py start
    python3 advanced_scheduler.py status
    python3 advanced_scheduler.py stop
    python3 advanced_scheduler.py run --script twitter
"""

import os
import sys
import time
import signal
import logging
import subprocess
import datetime
import json
import smtplib
import socket
import fcntl
import resource
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import pytz
import psutil
from dotenv import load_dotenv
import argparse

# Load environment variables
load_dotenv()

# Script Configuration
CONFIG = {
    "log_dir": "logs",
    "pid_file": "scheduler.pid",
    "status_file": "scheduler_status.json",
    "lock_file": "scheduler.lock",
    "max_retries": 3,
    "retry_delay": 300,  # 5 minutes
    "email_notifications": True,
    "sast_timezone": pytz.timezone("Africa/Johannesburg"),
    "scripts": {
        "twitter": {
            "path": "twitter_content_generator.py",
            "times_per_day": 4,
            "max_runtime": 300,  # 5 minutes
            "priority": "high",
        },
        "facebook": {
            "path": "facebook_content_generator.py",
            "times_per_day": 3,
            "max_runtime": 300,  # 5 minutes
            "priority": "medium",
        },
        "linkedin": {
            "path": "linkedin_content_generator.py",
            "times_per_day": 3,
            "max_runtime": 300,  # 5 minutes
            "priority": "medium",
        },
    },
}

# Email configuration
EMAIL_CONFIG = {
    "enabled": os.getenv("EMAIL_NOTIFICATIONS", "False").lower() == "true",
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "username": os.getenv("EMAIL_USERNAME", ""),
    "password": os.getenv("EMAIL_PASSWORD", ""),
    "from_email": os.getenv("FROM_EMAIL", "alerts@writenowagency.com"),
    "to_email": os.getenv("TO_EMAIL", "admin@writenowagency.com"),
}


# Setup logging
def setup_logging():
    """Set up rotating file logging"""
    if not os.path.exists(CONFIG["log_dir"]):
        os.makedirs(CONFIG["log_dir"])

    log_file = os.path.join(CONFIG["log_dir"], "scheduler.log")

    # Create a rotating file handler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=10
    )
    file_handler.setLevel(logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Get the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Check if handlers exist to avoid duplicates
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


logger = setup_logging()


class SchedulerLock:
    """File-based lock to ensure only one instance of the scheduler runs"""

    def __init__(self, lock_file):
        self.lock_file = lock_file
        self.lock_fd = None

    def acquire(self):
        """Acquire a lock, returns True if successful, False otherwise"""
        try:
            self.lock_fd = open(self.lock_file, "w")
            fcntl.lockf(self.lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.lock_fd.write(str(os.getpid()))
            self.lock_fd.flush()
            return True
        except IOError:
            # Another instance is already running
            if self.lock_fd:
                self.lock_fd.close()
                self.lock_fd = None
            return False

    def release(self):
        """Release the lock"""
        if self.lock_fd:
            fcntl.lockf(self.lock_fd, fcntl.LOCK_UN)
            self.lock_fd.close()
            self.lock_fd = None
            try:
                os.remove(self.lock_file)
            except OSError:
                pass


class ContentGeneratorScheduler:
    """Advanced scheduler for managing social media content generation"""

    def __init__(self):
        self.scheduler = BackgroundScheduler(timezone=CONFIG["sast_timezone"])
        self.job_status = {}
        self.running_jobs = {}
        self.lock = SchedulerLock(CONFIG["lock_file"])

        # Register listeners for job execution and errors
        self.scheduler.add_listener(self._job_executed_listener, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error_listener, EVENT_JOB_ERROR)

        # Create status file if it doesn't exist
        self._init_status_file()

    def _init_status_file(self):
        """Initialize the status file with default values"""
        if not os.path.exists(CONFIG["status_file"]):
            default_status = {
                "last_update": datetime.datetime.now().isoformat(),
                "is_running": False,
                "jobs": {
                    script_name: {
                        "last_run": None,
                        "next_run": None,
                        "success_count": 0,
                        "failure_count": 0,
                        "last_status": None,
                        "last_error": None,
                    }
                    for script_name in CONFIG["scripts"]
                },
                "system_stats": {"cpu_usage": 0, "memory_usage": 0, "uptime": 0},
            }
            with open(CONFIG["status_file"], "w") as f:
                json.dump(default_status, f, indent=2)
            logger.info(f"Initialized status file at {CONFIG['status_file']}")

    def _calculate_job_times(self, script_name):
        """Calculate high-traffic posting times for each social platform"""
        script_config = CONFIG["scripts"][script_name]

        # High-traffic times for different platforms (in SAST timezone)
        high_traffic_times = {
            "twitter": [
                # Morning commute + lunch break + evening commute + night browsing
                datetime.time(hour=7, minute=30),  # Morning commute/breakfast
                datetime.time(hour=12, minute=15),  # Lunch break
                datetime.time(hour=17, minute=45),  # Evening commute
                datetime.time(hour=20, minute=30),  # Evening leisure time
            ],
            "facebook": [
                # Lunch time + after work + evening prime time
                datetime.time(hour=13, minute=0),  # Lunch break
                datetime.time(hour=18, minute=30),  # After work hours
                datetime.time(hour=20, minute=0),  # Evening prime time
            ],
            "linkedin": [
                # Early business hours + lunch + end of workday
                datetime.time(hour=8, minute=45),  # Start of business day
                datetime.time(hour=12, minute=30),  # Lunch break
                datetime.time(hour=16, minute=45),  # End of business day
            ],
        }

        # Select times for this specific platform
        if script_name in high_traffic_times:
            # Get pre-defined high-traffic times
            run_times = high_traffic_times[script_name]

            # Ensure we have the correct number of times
            if len(run_times) > script_config["times_per_day"]:
                # If we have more times than needed, take the first N
                run_times = run_times[: script_config["times_per_day"]]
            elif len(run_times) < script_config["times_per_day"]:
                # If we have fewer times than needed, add some
                # (This shouldn't happen with our current configuration)
                day_minutes = 24 * 60
                interval_minutes = day_minutes // script_config["times_per_day"]

                while len(run_times) < script_config["times_per_day"]:
                    # Find a time that's not too close to existing times
                    for hour in range(8, 21):  # Business hours
                        new_time = datetime.time(hour=hour, minute=0)
                        # Check if this time is far enough from existing times
                        if all(
                            abs((hour * 60) - (t.hour * 60 + t.minute)) > 60
                            for t in run_times
                        ):
                            run_times.append(new_time)
                            break

                    # If we still need more times, just add them at regular intervals
                    if len(run_times) < script_config["times_per_day"]:
                        minutes_from_midnight = len(run_times) * interval_minutes
                        hour = minutes_from_midnight // 60
                        minute = minutes_from_midnight % 60
                        run_times.append(datetime.time(hour=hour, minute=minute))
        else:
            # Fallback to evenly distributed times (shouldn't be needed)
            day_minutes = 24 * 60
            interval_minutes = day_minutes // script_config["times_per_day"]

            # Offset between scripts to avoid concurrent execution
            offset = {"twitter": 0, "facebook": 20, "linkedin": 40}.get(script_name, 0)

            run_times = []
            for i in range(script_config["times_per_day"]):
                # Calculate minutes from midnight
                minutes_from_midnight = (i * interval_minutes + offset) % day_minutes
                hour = minutes_from_midnight // 60
                minute = minutes_from_midnight % 60

                # Create a time object
                run_time = datetime.time(hour=hour, minute=minute)
                run_times.append(run_time)

        logger.info(
            f"High-traffic posting times for {script_name}: {', '.join([f'{t.hour}:{t.minute:02d}' for t in run_times])}"
        )
        return run_times

    def _job_executed_listener(self, event):
        """Handle successful job execution events"""
        job_id = event.job_id
        if job_id in self.job_status:
            script_name = self.job_status[job_id]["script_name"]
            logger.info(f"Job completed successfully: {script_name}")

            # Update status
            self._update_job_status(script_name, True)

    def _job_error_listener(self, event):
        """Handle job error events"""
        job_id = event.job_id
        if job_id in self.job_status:
            script_name = self.job_status[job_id]["script_name"]
            exception = event.exception
            logger.error(f"Job failed: {script_name}, Error: {exception}")

            # Update status
            self._update_job_status(script_name, False, str(exception))

            # Check if we should retry
            retry_count = self.job_status[job_id].get("retries", 0)
            if retry_count < CONFIG["max_retries"]:
                self._schedule_retry(script_name, retry_count + 1)
            else:
                logger.error(f"Max retries reached for {script_name}")
                self._send_alert(
                    f"Max retries reached for {script_name}", str(exception)
                )

    def _update_job_status(self, script_name, success, error=None):
        """Update the status file with latest job execution info"""
        try:
            # Load current status
            with open(CONFIG["status_file"], "r") as f:
                status = json.load(f)

            # Update the job status
            now = datetime.datetime.now().isoformat()
            job_info = status["jobs"][script_name]
            job_info["last_run"] = now

            if success:
                job_info["success_count"] += 1
                job_info["last_status"] = "success"
            else:
                job_info["failure_count"] += 1
                job_info["last_status"] = "failure"
                job_info["last_error"] = error

            # Get next run time if scheduler is running
            if self.scheduler.running:
                next_run = None
                for job_id, details in self.job_status.items():
                    if details["script_name"] == script_name:
                        job = self.scheduler.get_job(job_id)
                        if job and job.next_run_time:
                            next_run = job.next_run_time.isoformat()
                            break

                job_info["next_run"] = next_run

            # Update system stats
            status["system_stats"] = self._get_system_stats()
            status["last_update"] = now
            status["is_running"] = True

            # Save updated status
            with open(CONFIG["status_file"], "w") as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating job status: {e}")

    def _schedule_retry(self, script_name, retry_count):
        """Schedule a retry for a failed job"""
        logger.info(f"Scheduling retry {retry_count} for {script_name}")

        # Schedule retry after delay
        delay = CONFIG["retry_delay"] * (2 ** (retry_count - 1))  # Exponential backoff
        run_date = datetime.datetime.now() + datetime.timedelta(seconds=delay)

        job = self.scheduler.add_job(
            func=self._run_script,
            trigger="date",
            run_date=run_date,
            args=[script_name, True, retry_count],
            id=f"{script_name}_retry_{retry_count}_{int(time.time())}",
        )

        # Track job status
        self.job_status[job.id] = {
            "script_name": script_name,
            "retries": retry_count,
            "is_retry": True,
        }

        logger.info(f"Retry for {script_name} scheduled at {run_date}")

    def _send_alert(self, subject, message):
        """Send email alert for critical issues"""
        if not EMAIL_CONFIG["enabled"]:
            logger.info("Email notifications disabled. Would have sent: " + subject)
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = EMAIL_CONFIG["from_email"]
            msg["To"] = EMAIL_CONFIG["to_email"]
            msg["Subject"] = f"WriteNow Scheduler Alert: {subject}"

            # Add hostname to message
            hostname = socket.gethostname()
            full_message = f"Host: {hostname}\n\n{message}"

            msg.attach(MIMEText(full_message, "plain"))

            server = smtplib.SMTP(
                EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]
            )
            server.starttls()
            server.login(EMAIL_CONFIG["username"], EMAIL_CONFIG["password"])
            server.send_message(msg)
            server.quit()

            logger.info(f"Alert email sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _get_system_stats(self):
        """Get system resource statistics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # System uptime
            uptime = time.time() - psutil.boot_time()

            return {
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "uptime": uptime,
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"cpu_usage": 0, "memory_usage": 0, "uptime": 0}

    def _run_script(self, script_name, is_retry=False, retry_count=0):
        """Execute a content generator script"""
        script_config = CONFIG["scripts"][script_name]
        script_path = script_config["path"]

        logger.info(
            f"Running {script_name} script{' (retry attempt '+str(retry_count)+')' if is_retry else ''}"
        )

        # Set resource limits
        def preexec_fn():
            # Set process nice value (priority)
            nice_value = 10  # Default
            if script_config["priority"] == "high":
                nice_value = 0
            elif script_config["priority"] == "medium":
                nice_value = 10
            else:  # low
                nice_value = 19

            os.nice(nice_value)

            # Set CPU time limit
            resource.setrlimit(
                resource.RLIMIT_CPU,
                (script_config["max_runtime"], script_config["max_runtime"]),
            )

        try:
            # Start the process
            process = subprocess.Popen(
                ["python3", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=preexec_fn,
            )

            # Store in running jobs
            self.running_jobs[script_name] = process

            # Set a timeout for the process
            timeout = script_config["max_runtime"]
            start_time = time.time()

            stdout, stderr = b"", b""

            # Poll the process instead of waiting indefinitely
            while process.poll() is None:
                # Check if we've exceeded the timeout
                if time.time() - start_time > timeout:
                    logger.warning(
                        f"{script_name} exceeded timeout of {timeout} seconds, terminating..."
                    )
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    stdout, stderr = process.communicate()
                    error_msg = (
                        f"Process killed after exceeding {timeout} seconds timeout"
                    )
                    self._update_job_status(script_name, False, error_msg)
                    return

                time.sleep(1)

            # Process completed, get output
            stdout, stderr = process.communicate()

            # Check return code
            if process.returncode != 0:
                logger.error(
                    f"{script_name} failed with return code {process.returncode}"
                )
                error_output = stderr.decode("utf-8", errors="replace")
                logger.error(f"Error output: {error_output}")
                self._update_job_status(script_name, False, error_output)

                # Send notification for non-retry failures
                if not is_retry:
                    self._send_alert(
                        f"{script_name} script failed",
                        f"Return code: {process.returncode}\nError: {error_output}",
                    )
                return

            # Process successful logs
            output = stdout.decode("utf-8", errors="replace")
            logger.info(f"{script_name} completed successfully")

            # Update internal tracking
            if script_name in self.running_jobs:
                del self.running_jobs[script_name]

            # If this was a retry, log success after retry
            if is_retry:
                logger.info(f"Retry {retry_count} for {script_name} succeeded")

        except Exception as e:
            logger.error(f"Error running {script_name}: {e}")
            self._update_job_status(script_name, False, str(e))

            # Send notification for non-retry failures
            if not is_retry:
                self._send_alert(
                    f"Error executing {script_name}", f"Exception: {str(e)}"
                )

    def schedule_jobs(self):
        """Schedule all content generator scripts"""
        for script_name, config in CONFIG["scripts"].items():
            logger.info(
                f"Scheduling {script_name} to run {config['times_per_day']} times per day"
            )

            # Calculate run times
            run_times = self._calculate_job_times(script_name)

            # Schedule each run time
            for i, run_time in enumerate(run_times):
                job = self.scheduler.add_job(
                    func=self._run_script,
                    trigger="cron",
                    hour=run_time.hour,
                    minute=run_time.minute,
                    args=[script_name],
                    id=f"{script_name}_{i}",
                )

                # Track job status
                self.job_status[job.id] = {"script_name": script_name, "run_index": i}

                # Log the scheduling
                logger.info(
                    f"Scheduled {script_name} to run at {run_time.hour}:{run_time.minute:02d}"
                )

    def _update_next_run_times(self):
        """Update the next run times in the status file"""
        # Only update next run times after scheduler has started
        if not self.scheduler.running:
            return

        try:
            # Load current status
            with open(CONFIG["status_file"], "r") as f:
                status = json.load(f)

            for script_name in CONFIG["scripts"].keys():
                # Find all jobs for this script
                next_run = None
                for job_id, details in self.job_status.items():
                    if details["script_name"] == script_name:
                        job = self.scheduler.get_job(job_id)
                        if job and job.next_run_time:
                            job_next_run = job.next_run_time
                            # Update if this job runs sooner than current next_run
                            if next_run is None or job_next_run < next_run:
                                next_run = job_next_run

                # Update the status
                if next_run:
                    status["jobs"][script_name]["next_run"] = next_run.isoformat()

            # Save updated status
            with open(CONFIG["status_file"], "w") as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Error updating next run times: {e}")

    def start(self):
        """Start the scheduler"""
        try:
            # Try to acquire the lock
            if not self.lock.acquire():
                logger.error("Another instance is already running. Exiting.")
                sys.exit(1)

            # Write PID file
            with open(CONFIG["pid_file"], "w") as f:
                f.write(str(os.getpid()))

            logger.info("Starting content generator scheduler")

            # Update status file
            with open(CONFIG["status_file"], "r") as f:
                status = json.load(f)
            status["is_running"] = True
            status["last_update"] = datetime.datetime.now().isoformat()
            with open(CONFIG["status_file"], "w") as f:
                json.dump(status, f, indent=2)

            # Schedule all jobs
            self.schedule_jobs()

            # Start the scheduler
            self.scheduler.start()
            logger.info("Scheduler started")

            # Update next run times after scheduler has started
            self._update_next_run_times()

            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGTERM, self._handle_signals)
            signal.signal(signal.SIGINT, self._handle_signals)

            # Run continuously until interrupted
            while True:
                time.sleep(60)
                self._check_health()

        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
            self._cleanup()
            sys.exit(1)

    def _check_health(self):
        """Periodically check health of the system and the scheduler"""
        try:
            # Update system stats
            with open(CONFIG["status_file"], "r") as f:
                status = json.load(f)

            status["system_stats"] = self._get_system_stats()
            status["last_update"] = datetime.datetime.now().isoformat()

            # Check for stalled jobs
            for script_name, process in list(self.running_jobs.items()):
                if process.poll() is not None:  # Process has terminated
                    logger.warning(f"Process for {script_name} terminated unexpectedly")
                    del self.running_jobs[script_name]

            # Save updated status
            with open(CONFIG["status_file"], "w") as f:
                json.dump(status, f, indent=2)

        except Exception as e:
            logger.error(f"Error in health check: {e}")

    def _handle_signals(self, signum, frame):
        """Handle termination signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()

    def stop(self):
        """Stop the scheduler and cleanup"""
        logger.info("Stopping content generator scheduler")

        # Shutdown the scheduler
        if hasattr(self, "scheduler") and self.scheduler.running:
            self.scheduler.shutdown()

        # Kill any running processes
        for script_name, process in self.running_jobs.items():
            logger.info(f"Terminating running process for {script_name}")
            try:
                process.terminate()
                time.sleep(1)
                if process.poll() is None:  # If still running
                    process.kill()
            except Exception as e:
                logger.error(f"Error terminating process: {e}")

        # Update status file
        try:
            with open(CONFIG["status_file"], "r") as f:
                status = json.load(f)
            status["is_running"] = False
            status["last_update"] = datetime.datetime.now().isoformat()
            with open(CONFIG["status_file"], "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Error updating status file on shutdown: {e}")

        self._cleanup()
        logger.info("Scheduler stopped")

    def _cleanup(self):
        """Clean up resources"""
        # Release the lock
        self.lock.release()

        # Remove PID file
        try:
            if os.path.exists(CONFIG["pid_file"]):
                os.remove(CONFIG["pid_file"])
        except Exception as e:
            logger.error(f"Error removing PID file: {e}")

    def check_status(self):
        """Check and display the current status of the scheduler"""
        try:
            # Check if scheduler is running
            is_running = False
            if os.path.exists(CONFIG["pid_file"]):
                with open(CONFIG["pid_file"], "r") as f:
                    pid = int(f.read().strip())
                try:
                    # Check if process with this PID exists
                    process = psutil.Process(pid)
                    is_running = True
                    print(f"Scheduler is running (PID: {pid})")
                except psutil.NoSuchProcess:
                    print("Scheduler PID file exists but process is not running")
            else:
                print("Scheduler is not running")

            # Load status file
            if os.path.exists(CONFIG["status_file"]):
                with open(CONFIG["status_file"], "r") as f:
                    status = json.load(f)

                print("\nLast update:", status["last_update"])
                print("System stats:")
                stats = status["system_stats"]
                print(f"  - CPU: {stats['cpu_usage']:.1f}%")
                print(f"  - Memory: {stats['memory_usage']:.1f}%")

                print("\nJob status:")
                for script_name, job_info in status["jobs"].items():
                    print(f"\n{script_name.upper()}:")
                    print(f"  - Last run: {job_info['last_run'] or 'Never'}")
                    print(f"  - Next run: {job_info['next_run'] or 'Not scheduled'}")
                    print(f"  - Success count: {job_info['success_count']}")
                    print(f"  - Failure count: {job_info['failure_count']}")
                    print(f"  - Last status: {job_info['last_status'] or 'N/A'}")

                    if job_info["last_status"] == "failure" and job_info["last_error"]:
                        print(f"  - Last error: {job_info['last_error']}")
            else:
                print("Status file not found")

            return is_running

        except Exception as e:
            print(f"Error checking status: {e}")
            return False

    def run_once(self, script_name):
        """Run a specified script immediately once"""
        if script_name not in CONFIG["scripts"]:
            print(f"Invalid script name: {script_name}")
            return False

        print(f"Running {script_name} immediately...")
        self._run_script(script_name)
        print("Done")
        return True


def start_scheduler():
    """Start the scheduler as a daemon process"""
    print("Starting WriteNow content generator scheduler...")

    # Create scheduler instance
    scheduler = ContentGeneratorScheduler()

    # Start in current process
    scheduler.start()


def check_scheduler_status():
    """Check the current status of the scheduler"""
    scheduler = ContentGeneratorScheduler()
    return scheduler.check_status()


def stop_scheduler():
    """Stop the running scheduler"""
    print("Stopping WriteNow content generator scheduler...")

    # Check if PID file exists
    if os.path.exists(CONFIG["pid_file"]):
        try:
            with open(CONFIG["pid_file"], "r") as f:
                pid = int(f.read().strip())

            # Send SIGTERM to the process
            os.kill(pid, signal.SIGTERM)
            print(f"Sent termination signal to process {pid}")

            # Wait a moment to see if it terminates
            for _ in range(5):
                time.sleep(1)
                try:
                    # Check if process still exists
                    os.kill(pid, 0)
                except OSError:
                    print("Scheduler stopped successfully")
                    return True

            # If we get here, it didn't stop
            print("Scheduler did not stop gracefully, forcing termination...")
            os.kill(pid, signal.SIGKILL)
            print("Scheduler terminated")

        except Exception as e:
            print(f"Error stopping scheduler: {e}")
    else:
        print("Scheduler does not appear to be running (no PID file)")

    # Clean up PID and lock files
    for file in [CONFIG["pid_file"], CONFIG["lock_file"]]:
        if os.path.exists(file):
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error removing {file}: {e}")

    return True


def run_script_once(script_name):
    """Run a specified script once immediately"""
    scheduler = ContentGeneratorScheduler()
    return scheduler.run_once(script_name)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="WriteNow Agency Content Generator Scheduler"
    )
    parser.add_argument(
        "command", choices=["start", "stop", "status", "run"], help="Command to execute"
    )
    parser.add_argument(
        "--script",
        choices=list(CONFIG["scripts"].keys()),
        help="Script to run (only applicable with run command)",
    )
    parser.add_argument(
        "--nohup",
        action="store_true",
        help="Run in background with nohup (only applicable with start command)",
    )

    args = parser.parse_args()

    if args.command == "start":
        if args.nohup:
            print("Starting scheduler with nohup...")

            # Prepare nohup command
            nohup_log = os.path.join(CONFIG["log_dir"], "nohup.out")
            cmd = f"nohup python3 {__file__} start > {nohup_log} 2>&1 &"

            # Execute the command
            subprocess.Popen(cmd, shell=True)
            print(f"Scheduler started in background. Check {nohup_log} for output.")
        else:
            start_scheduler()

    elif args.command == "stop":
        stop_scheduler()

    elif args.command == "status":
        check_scheduler_status()

    elif args.command == "run":
        if not args.script:
            print("Error: --script is required with the run command")
            print(f"Available scripts: {', '.join(CONFIG['scripts'].keys())}")
            return

        run_script_once(args.script)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
