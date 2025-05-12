CREATE DATABASE rag WITH OWNER postgres;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
  chunk_id  text PRIMARY KEY,
  pdf_name  text,
  text      text,
  embedding vector(4096),
  doc_type text,
  case_id text,
  extra jsonb
);

CREATE TABLE IF NOT EXISTS kg_loaded(source_id text primary key);
-- Create job_queue table
CREATE TABLE IF NOT EXISTS job_queue (
    job_id SERIAL PRIMARY KEY,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, processing, completed, failed
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    job_data JSONB NOT NULL,  -- Store job parameters as JSON
    result JSONB,             -- Store job results as JSON
    worker_id VARCHAR(100),   -- ID of worker processing this job
    error_message TEXT        -- Error message if job failed
);

-- Create index for faster status queries
CREATE INDEX IF NOT EXISTS idx_job_queue_status ON job_queue(status);

-- Create notification function
CREATE OR REPLACE FUNCTION notify_job_change()
RETURNS TRIGGER AS $$
BEGIN
    -- Notify about job changes
    PERFORM pg_notify('job_queue_channel', json_build_object(
        'operation', TG_OP,
        'job_id', NEW.job_id,
        'status', NEW.status
    )::text);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for job changes
DROP TRIGGER IF EXISTS job_queue_notify_trigger ON job_queue;
CREATE TRIGGER job_queue_notify_trigger
AFTER INSERT OR UPDATE ON job_queue
FOR EACH ROW EXECUTE FUNCTION notify_job_change();

-- Create view for queue statistics
CREATE OR REPLACE VIEW queue_stats AS
SELECT 
    COUNT(*) FILTER (WHERE status = 'pending') AS pending_jobs,
    COUNT(*) FILTER (WHERE status = 'processing') AS processing_jobs,
    COUNT(*) FILTER (WHERE status = 'completed') AS completed_jobs,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_jobs
FROM job_queue;