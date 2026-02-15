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
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from utils.logger import logger


# Directory for generated ICS event files
EVENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "events")
os.makedirs(EVENTS_DIR, exist_ok=True)


class RichResponseProcessor:
    """Post-processes LLM responses to extract rich content blocks.
    
    Rich content types:
    - image: Inline image from the archive (WhatsApp media)
    - ics_event: Calendar event for download
    - buttons: Clickable options for disambiguation/clarification
    """
    
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
        
        # 1. Extract inline images from source nodes
        images = self._extract_images(source_nodes or [])
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
    ) -> List[Dict[str, Any]]:
        """Extract image content blocks from source nodes that have media.
        
        Scans source nodes for entries with has_media=True and a valid
        media_path. Returns image blocks with serving URLs.
        
        Args:
            source_nodes: List of NodeWithScore from the retriever
            
        Returns:
            List of image rich content blocks
        """
        images: List[Dict[str, Any]] = []
        seen_paths: set = set()
        
        for node_with_score in source_nodes:
            node = getattr(node_with_score, 'node', None)
            if not node:
                continue
            
            metadata = getattr(node, 'metadata', {})
            
            # Skip non-media nodes
            has_media = metadata.get('has_media', False)
            if not has_media:
                continue
            
            media_path = metadata.get('media_path', '')
            if not media_path:
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
                logger.debug(f"Media file not found on disk: {full_path}")
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
            
            images.append({
                "type": "image",
                "url": f"/media/images/{filename}",
                "alt": caption,
                "caption": caption,
            })
        
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
