#!/usr/bin/env python3
import os
import time
import json
import select
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
from main import app
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Get database connection settings from environment variables
DB_HOST = os.getenv("DB_HOST", "postgres")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "rag")
DB_USER = os.getenv("DB_USER", "demo")
DB_PASSWORD = os.getenv("DB_PASSWORD", "demo1234")
# How long to wait (seconds) with no NOTIFY before assuming the queue is empty
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "30"))

DB_DSN = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def connect():
    logger.info(f"Connecting to {DB_DSN}")
    conn = psycopg2.connect(DB_DSN)
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("LISTEN new_jobs;")
    print("Worker: listening on channel 'new_jobs'")
    return conn, cur

def fetch_and_lock_job(conn):
    """Attempt to claim one pending job. Return the job dict or None."""
    logger.info("Fetching and locking job")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("BEGIN;")
        cur.execute("""
            SELECT job_id, job_data
              FROM job_queue
             WHERE status = 'pending'
             ORDER BY created_at
             LIMIT 1
            FOR UPDATE SKIP LOCKED
        """)
        job = cur.fetchone()
        logger.info(f"Fetched job: {job}")

        if not job:
            cur.execute("ROLLBACK;")
            return None

        cur.execute(
            "UPDATE jobs SET status = 'in_progress', updated_at = NOW() WHERE id = %(id)s;",
            {"id": job["id"]}
        )
        cur.execute("COMMIT;")
        return job

def complete_job(conn, job_id, result):
    logger.info(f"Completing job {job_id} with result {result}")
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE job_queue
               SET status = 'completed'
                 , job_data = %s
                 , updated_at = NOW()
             WHERE job_id = %s
        """, (json.dumps(result), job_id))
        conn.commit()

def process_job(job):
    """Placeholder for your AI-agent work."""
    logger.info(f"Processing job {job['id']} payload={job['payload']}")
    # --- your AI logic here! ---
    result = app.invoke({"rfp_text": "", "questions": [], "answers": {}})
    return {"message": f"Generated document for job {job['id']}"}

def main():
    conn, listen_cur = connect()
    last_activity = time.time()
    
    # 1) On startup, catch up any jobs already pending
    job = fetch_and_lock_job(conn)
    if job:
        result = process_job(job)
        complete_job(conn, job["id"], result)
        last_activity = time.time()

    # 2) Enter notification loop
    while True:
        # Wait for NOTIFY or timeout
        timeout = max(0, IDLE_TIMEOUT - (time.time() - last_activity))
        if timeout <= 0:
            print("Worker: idle timeout reached, exiting.")
            break

        ready, _, _ = select.select([conn], [], [], timeout)
        if not ready:
            # no notifications in timeout period
            print("Worker: no notify within idle timeout, exiting.")
            break

        # got a notification
        conn.poll()
        for notify in conn.notifies:
            print(f"Worker: got NOTIFY: {notify.channel} payload={notify.payload}")
            # whenever you get ANY notify, try to fetch one job
            job = fetch_and_lock_job(conn)
            if job:
                result = process_job(job)
                complete_job(conn, job["id"], result)
                last_activity = time.time()
        # clear notifications so we don't re-process the same ones
        conn.notifies.clear()

    listen_cur.close()
    conn.close()
    print("Worker: shutdown complete.")

if __name__ == "__main__":
    main()
