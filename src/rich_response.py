"""Rich response post-processor for Lucy's AI responses.

Extracts structured rich content from the LLM's text answer and source nodes:
- Inline images from source nodes with media attachments
- ICS calendar events from [CREATE_EVENT] markers in the answer
- Interactive buttons from disambiguation/clarification patterns

Usage:
    >>> from rich_response import RichResponseProcessor
    >>> processor = RichResponseProcessor()
    >>> answer, rich_content = processor.process(raw_answer, source_nodes)
"""

import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from utils.logger import logger


# Directory for generated ICS event files
EVENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "events")
os.makedirs(EVENTS_DIR, exist_ok=True)


# Roles assigned to nodes that serve as LLM context but should not appear
# as user-visible sources.  Set in llamaindex_rag.py during retrieval.
_CONTEXT_ONLY_ROLES: Set[str] = {
    "context_expansion",
    "recency_supplement",
    "asset_neighborhood",
}


class RichResponseProcessor:
    """Post-processes LLM responses to extract rich content blocks.
    
    Rich content types:
    - image: Inline image from the archive (WhatsApp media)
    - ics_event: Calendar event for download
    - buttons: Clickable options for disambiguation/clarification
    """
    
    # =========================================================================
    # Source Relevance Filtering
    # =========================================================================
    
    def filter_sources_for_display(
        self,
        source_nodes: List[Any],
        answer: str,
        min_score: float = 0.5,
        max_count: int = 8,
        answer_filter: bool = True,
    ) -> List[Any]:
        """Filter source nodes to only show relevant ones to the user.
        
        The retriever intentionally casts a wide net for the LLM context
        (context expansion, recency supplements, asset neighborhood).
        This method prunes those internal-context nodes so the user only
        sees sources that meaningfully contributed to the answer.
        
        Filtering pipeline (applied in order):
        1. Exclude system/placeholder nodes
        2. Exclude context-only roles (context_expansion, recency_supplement, etc.)
        3. Score threshold — drop nodes below ``min_score``
        4. Answer-relevance — keep only sources referenced in the answer text
        5. Max count cap
        
        Args:
            source_nodes: Raw source nodes from the chat engine response
            answer: The LLM's generated answer text
            min_score: Minimum score for a source to be displayed (default 0.5)
            max_count: Maximum number of sources to show (default 8)
            answer_filter: Whether to apply answer-relevance filtering (default True)
            
        Returns:
            Filtered list of source nodes suitable for user display
        """
        if not source_nodes:
            return []
        
        filtered = []
        answer_lower = answer.lower() if answer else ""
        
        for nws in source_nodes:
            node = getattr(nws, "node", None)
            if not node:
                continue
            
            metadata = getattr(node, "metadata", {})
            score = getattr(nws, "score", None)
            
            # Layer 1: Skip system/placeholder nodes
            if metadata.get("source") == "system":
                continue
            
            # Layer 2: Skip context-only roles
            source_role = metadata.get("source_role", "")
            if source_role in _CONTEXT_ONLY_ROLES:
                continue
            
            # Layer 3: Score threshold
            # Entity store facts (score=1.0) always pass.
            # Primary search + reranked results typically score > 0.5.
            if score is not None and score < min_score:
                continue
            
            # Layer 4: Answer-relevance check
            # A source is relevant if:
            #   a) It's an entity_store fact (always relevant — factual answers)
            #   b) Its sender name appears in the answer
            #   c) Its chat_name appears in the answer
            #   d) A meaningful content snippet (>20 chars) appears in the answer
            if answer_filter and answer_lower:
                source_type = metadata.get("source", "")
                
                # Entity store facts are always relevant
                if source_type == "entity_store":
                    filtered.append(nws)
                    continue
                
                # Check if sender or chat_name is referenced in the answer
                sender = metadata.get("sender", "")
                chat_name = metadata.get("chat_name", "")
                is_referenced = False
                
                if sender and len(sender) >= 2 and sender.lower() in answer_lower:
                    is_referenced = True
                elif chat_name and len(chat_name) >= 2 and chat_name.lower() in answer_lower:
                    is_referenced = True
                else:
                    # Check if distinctive content from the source appears in the answer
                    # Use the first meaningful sentence (>20 chars) from the source text
                    node_text = getattr(node, "text", "") or ""
                    # Extract lines that look like content (not headers/metadata)
                    for line in node_text.split("\n"):
                        line = line.strip()
                        if len(line) > 20 and not line.startswith(("[", "Entity Store")):
                            # Check if a substantial substring appears in the answer
                            # Use first 60 chars of the line as a fingerprint
                            snippet = line[:60].lower()
                            if snippet in answer_lower:
                                is_referenced = True
                                break
                
                if not is_referenced:
                    continue
            
            filtered.append(nws)
        
        pre_cap = len(filtered)
        
        # Layer 5: Max count cap (keep highest-scored)
        if len(filtered) > max_count:
            # Sort by score descending, keep top N
            filtered.sort(
                key=lambda nws: getattr(nws, "score", 0.0) or 0.0,
                reverse=True,
            )
            filtered = filtered[:max_count]
        
        if len(source_nodes) != len(filtered):
            logger.info(
                f"Source display filter: {len(source_nodes)} → {len(filtered)} sources "
                f"(role={len(source_nodes) - pre_cap - (len(source_nodes) - len(filtered)) if pre_cap < len(source_nodes) else 0} excluded, "
                f"cap={pre_cap - len(filtered) if pre_cap > len(filtered) else 0} capped)"
            )
        
        return filtered
    
    def process(
        self,
        answer: str,
        source_nodes: Optional[List[Any]] = None,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Post-process an LLM answer to extract rich content blocks.
        
        Args:
            answer: The raw text answer from the LLM
            source_nodes: Source nodes from the RAG retriever (NodeWithScore list)
            
        Returns:
            Tuple of (cleaned_answer_text, rich_content_list)
        """
        rich_content: List[Dict[str, Any]] = []
        
        # 1. Extract inline images from source nodes (filtered by answer context)
        images = self._extract_images(source_nodes or [], answer)
        rich_content.extend(images)
        
        # 2. Extract and generate ICS calendar events
        answer, events = self._extract_ics_events(answer)
        rich_content.extend(events)
        
        # 3. Extract disambiguation/clarification buttons
        answer, buttons = self._extract_buttons(answer)
        rich_content.extend(buttons)
        
        return answer, rich_content
    
    # =========================================================================
    # Feature 1: Inline Images
    # =========================================================================
    
    def _extract_images(
        self,
        source_nodes: List[Any],
        answer: str = "",
    ) -> List[Dict[str, Any]]:
        """Extract image content blocks from source nodes that have media.
        
        Scans source nodes for entries with has_media=True and a valid
        media_path. When an LLM answer is provided, filters to only show
        images whose sender or chat name is mentioned in the answer text —
        so only images the LLM is actually discussing are displayed.
        
        Args:
            source_nodes: List of NodeWithScore from the retriever
            answer: The LLM's answer text (used to filter relevant images)
            
        Returns:
            List of image rich content blocks
        """
        # First pass: collect all valid media nodes with metadata
        all_candidates: List[Dict[str, Any]] = []
        seen_paths: set = set()
        
        for node_with_score in source_nodes:
            node = getattr(node_with_score, 'node', None)
            if not node:
                continue
            
            metadata = getattr(node, 'metadata', {})
            
            # Skip non-media nodes — handle both bool and stringified values
            has_media = metadata.get('has_media', False)
            if isinstance(has_media, str):
                has_media = has_media.lower() in ('true', '1', 'yes')
            if not has_media:
                continue
            
            media_path = metadata.get('media_path', '')
            if not media_path:
                logger.info(f"Source node has has_media=True but empty media_path (sender={metadata.get('sender', '?')})")
                continue
            
            # Deduplicate
            if media_path in seen_paths:
                continue
            seen_paths.add(media_path)
            
            # Verify the file exists on disk
            full_path = media_path
            if not os.path.isabs(full_path):
                full_path = os.path.join(
                    os.path.dirname(os.path.dirname(__file__)),
                    media_path,
                )
            
            if not os.path.exists(full_path):
                logger.warning(f"Media file not found on disk: {full_path} (media_path={media_path})")
                continue
            
            # Build the serving URL — filename relative to data/images/
            filename = os.path.basename(media_path)
            
            # Build caption from metadata
            sender = metadata.get('sender', 'Unknown')
            chat_name = metadata.get('chat_name', '')
            timestamp = metadata.get('timestamp')
            time_str = ''
            if timestamp:
                try:
                    from config import settings
                    tz_name = settings.get("timezone", "Asia/Jerusalem")
                    dt = datetime.fromtimestamp(int(timestamp), tz=ZoneInfo(tz_name))
                    time_str = dt.strftime(" on %d/%m/%Y %H:%M")
                except (ValueError, TypeError):
                    pass
            
            caption = f"Image from {sender}"
            if chat_name:
                caption += f" in {chat_name}"
            if time_str:
                caption += time_str
            
            all_candidates.append({
                "type": "image",
                "url": f"/media/images/{filename}",
                "alt": caption,
                "caption": caption,
                "_sender": sender,
                "_chat_name": chat_name,
            })
        
        if not all_candidates:
            return []
        
        # Second pass: filter to images referenced in the LLM answer
        if answer and len(all_candidates) > 1:
            answer_lower = answer.lower()
            referenced = []
            for img in all_candidates:
                sender = img.get("_sender", "")
                chat_name = img.get("_chat_name", "")
                # Check if the sender or chat name is mentioned in the answer
                if sender and sender.lower() in answer_lower:
                    referenced.append(img)
                elif chat_name and chat_name.lower() in answer_lower:
                    referenced.append(img)
            if referenced:
                all_candidates = referenced
                logger.info(
                    f"Image extraction: filtered to {len(referenced)} image(s) "
                    f"referenced in answer (from {len(seen_paths)} candidates)"
                )
        
        # Strip internal metadata keys before returning
        images = []
        for img in all_candidates:
            clean = {k: v for k, v in img.items() if not k.startswith("_")}
            images.append(clean)
        
        logger.info(f"Image extraction: {len(seen_paths)} media node(s) found, {len(images)} image(s) extracted")
        return images
    
    # =========================================================================
    # Feature 2: ICS Calendar Events
    # =========================================================================
    
    # Pattern to match [CREATE_EVENT]...[/CREATE_EVENT] blocks
    _EVENT_PATTERN = re.compile(
        r'\[CREATE_EVENT\]\s*\n(.*?)\n\s*\[/CREATE_EVENT\]',
        re.DOTALL | re.IGNORECASE,
    )
    
    def _extract_ics_events(
        self,
        answer: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract calendar event markers and generate ICS files.
        
        Parses [CREATE_EVENT]...[/CREATE_EVENT] blocks from the answer,
        generates .ics files, and returns event blocks for the UI.
        
        Args:
            answer: Raw LLM answer text
            
        Returns:
            Tuple of (cleaned_answer, event_blocks)
        """
        events: List[Dict[str, Any]] = []
        
        matches = list(self._EVENT_PATTERN.finditer(answer))
        if not matches:
            return answer, events
        
        for match in matches:
            block_text = match.group(1)
            event_data = self._parse_event_block(block_text)
            
            if not event_data.get('title') or not event_data.get('start'):
                logger.warning(f"Incomplete event block, skipping: {block_text[:100]}")
                continue
            
            # Generate ICS file
            try:
                ics_filename = self._generate_ics_file(event_data)
                
                events.append({
                    "type": "ics_event",
                    "title": event_data['title'],
                    "start": event_data['start'],
                    "end": event_data.get('end', ''),
                    "location": event_data.get('location', ''),
                    "description": event_data.get('description', ''),
                    "download_url": f"/media/events/{ics_filename}",
                })
            except Exception as e:
                logger.error(f"Failed to generate ICS file: {e}")
        
        # Remove event markers from the visible answer
        cleaned = self._EVENT_PATTERN.sub('', answer).strip()
        # Clean up multiple blank lines left by removal
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned, events
    
    def _parse_event_block(self, block_text: str) -> Dict[str, str]:
        """Parse key-value pairs from a CREATE_EVENT block.
        
        Expected format:
            title: Meeting with David
            start: 2026-02-16T10:00
            end: 2026-02-16T11:00
            location: Office
            description: Discuss project updates
        
        Args:
            block_text: The text inside [CREATE_EVENT]...[/CREATE_EVENT]
            
        Returns:
            Dict of event field names to values
        """
        data: Dict[str, str] = {}
        
        for line in block_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Match "key: value" pattern
            colon_idx = line.find(':')
            if colon_idx <= 0:
                continue
            
            key = line[:colon_idx].strip().lower()
            value = line[colon_idx + 1:].strip()
            
            if key in ('title', 'start', 'end', 'location', 'description'):
                data[key] = value
        
        return data
    
    def _generate_ics_file(self, event_data: Dict[str, str]) -> str:
        """Generate an ICS file from event data and save to disk.
        
        Args:
            event_data: Dict with title, start, end, location, description
            
        Returns:
            Generated filename (e.g., 'meeting-with-david-abc123.ics')
        """
        from icalendar import Calendar, Event
        
        cal = Calendar()
        cal.add('prodid', '-//Lucy AI Assistant//EN')
        cal.add('version', '2.0')
        cal.add('calscale', 'GREGORIAN')
        cal.add('method', 'PUBLISH')
        
        event = Event()
        event.add('summary', event_data['title'])
        
        # Parse start datetime
        start_dt = self._parse_datetime(event_data['start'])
        event.add('dtstart', start_dt)
        
        # Parse end datetime (default: 1 hour after start)
        if event_data.get('end'):
            end_dt = self._parse_datetime(event_data['end'])
        else:
            end_dt = start_dt + timedelta(hours=1)
        event.add('dtend', end_dt)
        
        if event_data.get('location'):
            event.add('location', event_data['location'])
        
        if event_data.get('description'):
            event.add('description', event_data['description'])
        
        # Add UID and timestamp
        event.add('uid', f"{uuid.uuid4()}@lucy-assistant")
        event.add('dtstamp', datetime.now(ZoneInfo("UTC")))
        
        cal.add_component(event)
        
        # Generate filename
        safe_title = re.sub(r'[^\w\s-]', '', event_data['title'].lower())
        safe_title = re.sub(r'[\s]+', '-', safe_title.strip())[:50]
        short_id = uuid.uuid4().hex[:8]
        filename = f"{safe_title}-{short_id}.ics"
        
        # Save to disk
        filepath = os.path.join(EVENTS_DIR, filename)
        with open(filepath, 'wb') as f:
            f.write(cal.to_ical())
        
        logger.info(f"Generated ICS file: {filepath}")
        return filename
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse a datetime string in various formats.
        
        Supports:
        - ISO 8601: 2026-02-16T10:00:00, 2026-02-16T10:00
        - Date + time: 2026-02-16 10:00
        - Date only: 2026-02-16 (treated as all-day, midnight)
        
        Args:
            dt_str: Datetime string to parse
            
        Returns:
            Timezone-aware datetime
        """
        from config import settings
        tz_name = settings.get("timezone", "Asia/Jerusalem")
        tz = ZoneInfo(tz_name)
        
        # Try various formats
        formats = [
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%d/%m/%Y %H:%M",
            "%d/%m/%Y",
        ]
        
        dt_str = dt_str.strip()
        
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                return dt.replace(tzinfo=tz)
            except ValueError:
                continue
        
        # Fallback: try parsing as-is with a broader approach
        raise ValueError(f"Could not parse datetime: {dt_str!r}")
    
    # =========================================================================
    # Feature 3: Disambiguation Buttons
    # =========================================================================
    
    # Pattern for numbered options: "1) Option text" or "1. Option text"
    _OPTION_PATTERN = re.compile(
        r'^\s*(\d+)\s*[)\.]\s*(.+?)$',
        re.MULTILINE,
    )
    
    # Disambiguation question indicators (Hebrew + English)
    _QUESTION_INDICATORS = [
        'which one',
        'who did you mean',
        'did you mean',
        'please clarify',
        'please specify',
        'which person',
        'לאיזה',
        'למי התכוונת',
        'התכוונת',
        'איזה',
        'מי מהם',
        'תבחר',
        'תבחרי',
        'באיזה',
    ]
    
    def _extract_buttons(
        self,
        answer: str,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Extract disambiguation/clarification buttons from the answer.
        
        Detects patterns where the LLM presents numbered options and asks
        the user to choose. Extracts those into clickable button blocks.
        
        Args:
            answer: LLM answer text
            
        Returns:
            Tuple of (cleaned_answer, button_blocks)
        """
        buttons: List[Dict[str, Any]] = []
        
        # Check if the answer contains a disambiguation question
        answer_lower = answer.lower()
        has_question = any(
            indicator in answer_lower
            for indicator in self._QUESTION_INDICATORS
        )
        
        if not has_question:
            return answer, buttons
        
        # Find numbered options
        matches = list(self._OPTION_PATTERN.finditer(answer))
        
        if len(matches) < 2:
            # Need at least 2 options for disambiguation
            return answer, buttons
        
        # Extract options
        options: List[Dict[str, str]] = []
        option_lines_start = matches[0].start()
        option_lines_end = matches[-1].end()
        
        for match in matches:
            option_text = match.group(2).strip()
            # Clean trailing punctuation but preserve Hebrew/special chars
            option_text = option_text.rstrip('?？')
            options.append({
                "label": option_text,
                "value": option_text,
            })
        
        if not options:
            return answer, buttons
        
        # Extract the prompt text (everything before the numbered list)
        prompt_text = answer[:option_lines_start].strip()
        # Also get any text after the numbered list
        after_text = answer[option_lines_end:].strip()
        
        # If there's a question after the list, include it in the prompt
        if after_text:
            prompt_text = f"{prompt_text}\n{after_text}" if prompt_text else after_text
        
        # Clean prompt text — remove trailing colons
        prompt_text = prompt_text.rstrip(':').strip()
        
        buttons.append({
            "type": "buttons",
            "prompt": prompt_text,
            "options": options,
        })
        
        # Remove the numbered options from the visible answer
        # Keep the prompt text but replace options with empty space
        cleaned_lines = []
        option_numbers = {m.group(1) for m in matches}
        for line in answer.split('\n'):
            stripped = line.strip()
            # Skip lines that are numbered options
            is_option = False
            for num in option_numbers:
                if stripped.startswith(f"{num})") or stripped.startswith(f"{num}."):
                    is_option = True
                    break
            if not is_option:
                cleaned_lines.append(line)
        
        cleaned = '\n'.join(cleaned_lines).strip()
        # Clean multiple blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned, buttons
