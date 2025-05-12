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