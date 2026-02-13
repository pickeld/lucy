#!/bin/bash

# RAG Query Script
# Usage: ./scripts/query_rag.sh "your question here"
# Example: ./scripts/query_rag.sh "who said they would be late?"

# Configuration
HOST="${RAG_HOST:-http://localhost:8765}"
ENDPOINT="/rag/query"

# Check if question is provided
if [ -z "$1" ]; then
    echo "Usage: $0 \"your question\""
    echo "Example: $0 \"who mentioned the meeting?\""
    exit 1
fi

QUESTION="$1"

# Optional filters (can be passed as environment variables)
FILTER_CHAT="${FILTER_CHAT:-}"
FILTER_SENDER="${FILTER_SENDER:-}"
K="${K:-10}"

# Build JSON payload
if [ -n "$FILTER_CHAT" ] && [ -n "$FILTER_SENDER" ]; then
    PAYLOAD=$(cat <<EOF
{
    "question": "$QUESTION",
    "k": $K,
    "filter_chat_name": "$FILTER_CHAT",
    "filter_sender": "$FILTER_SENDER"
}
EOF
)
elif [ -n "$FILTER_CHAT" ]; then
    PAYLOAD=$(cat <<EOF
{
    "question": "$QUESTION",
    "k": $K,
    "filter_chat_name": "$FILTER_CHAT"
}
EOF
)
elif [ -n "$FILTER_SENDER" ]; then
    PAYLOAD=$(cat <<EOF
{
    "question": "$QUESTION",
    "k": $K,
    "filter_sender": "$FILTER_SENDER"
}
EOF
)
else
    PAYLOAD=$(cat <<EOF
{
    "question": "$QUESTION",
    "k": $K
}
EOF
)
fi

echo "🔍 Querying RAG..."
echo "Question: $QUESTION"
echo ""

# Make the request
RESPONSE=$(curl -s -X POST "${HOST}${ENDPOINT}" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

# Check if curl succeeded
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to connect to $HOST"
    exit 1
fi

# Check for error in response
ERROR=$(echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('error', ''))" 2>/dev/null)
if [ -n "$ERROR" ]; then
    echo "❌ Error: $ERROR"
    exit 1
fi

# Extract and print the answer
echo "📝 Answer:"
echo "$RESPONSE" | python3 -c "
import sys, json
d = json.load(sys.stdin)
answer = d.get('answer', 'No answer found')
# Handle case where answer might be a dict with 'text' key (Gemini format)
if isinstance(answer, dict):
    answer = answer.get('text', str(answer))
elif isinstance(answer, str) and answer.startswith(\"{'type':\"):
    try:
        parsed = eval(answer)
        answer = parsed.get('text', answer)
    except:
        pass
print(answer)
"

echo ""
echo "📊 Stats:"
echo "$RESPONSE" | python3 -c "import sys, json; d=json.load(sys.stdin); stats=d.get('stats', {}); print(f\"Total documents indexed: {stats.get('total_documents', 'N/A')}\")"
