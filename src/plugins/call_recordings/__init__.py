"""Call Recordings plugin for transcribing and indexing audio recordings.

Scans a local directory or Dropbox folder for audio call recordings,
transcribes them using local OpenAI Whisper, and indexes the transcriptions
into the Qdrant vector store for RAG retrieval.

Supports deduplication via SHA256 content hashing to avoid reprocessing
files that have already been indexed.
"""
