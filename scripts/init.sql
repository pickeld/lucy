-- LangGraph Database Initialization
-- This script ensures the database and schema are properly set up for LangGraph checkpointer

-- Create the langgraph database if it doesn't exist (should be created by env var)
-- The postgres user should already exist from the Docker image

-- Connect to the langgraph database
\c langgraph;

-- Create pgvector extension if needed (for future use)
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant necessary permissions to postgres user
GRANT ALL PRIVILEGES ON DATABASE langgraph TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO postgres;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO postgres;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO postgres;