#!/usr/bin/env python3
"""
Extract frames from YouTube video segments and generate:
- HTML lecture summary (for reading/summaries)
- Chunking-friendly JSON/JSONL (for vector DB ingestion; one chunk per segment).

Input: video_id (e.g. msHyYioAyNE) and transcript JSON. Outputs are under output/{video_id}/.
Uses yt-dlp (android client) to download video, ffmpeg to extract frames, OpenAI to refine content.
Video is not saved; only frames and generated files are kept.
"""

import json
import os
import shutil
import subprocess
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from openai import OpenAI

YOUTUBE_URL_PREFIX = "https://www.youtube.com/watch?v="


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or HH:MM:SS timestamp to seconds."""
    if timestamp is None:
        raise ValueError("Timestamp is None")
    timestamp = str(timestamp).strip()
    if not timestamp:
        raise ValueError("Timestamp is empty")
    parts = timestamp.split(':')
    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def safe_timestamp_to_seconds(timestamp, fallback: float) -> float:
    """Convert timestamp to seconds, returning fallback when missing/invalid."""
    try:
        return timestamp_to_seconds(timestamp)
    except (ValueError, TypeError):
        return fallback


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to MM:SS format."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def sanitize_filename(text: str) -> str:
    """Sanitize text to be used as a filename."""
    # Remove or replace invalid characters
    text = re.sub(r'[<>:"/\\|?*]', '_', text)
    text = re.sub(r'\s+', '_', text)
    return text[:100]  # Limit length


def download_full_video(youtube_url: str, output_dir: str) -> str:
    """
    Download the full video from YouTube using yt-dlp (android client strategy).
    Uses output template with %(ext)s so the file has the correct container extension
    (e.g. .webm or .mp4), avoiding "moov atom not found" when the stream is not MP4.

    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save the video (caller can delete after use)

    Returns:
        Path to the downloaded video file, or empty string on failure.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Let yt-dlp choose extension so container matches content (avoids ffmpeg "moov atom not found")
    output_template = os.path.join(output_dir, "video.%(ext)s")
    cmd = [
        'yt-dlp',
        '--no-playlist',
        '--format', 'best[height<=720]/best[height<=480]/best',
        '--extractor-args', 'youtube:player_client=android',
        '--output', output_template,
        youtube_url
    ]
    try:
        print("  Downloading video (yt-dlp, android client)...")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Find the downloaded file (e.g. video.webm or video.mp4)
        for name in os.listdir(output_dir):
            if name.startswith("video."):
                path = os.path.join(output_dir, name)
                if os.path.isfile(path):
                    print("  Video downloaded successfully.")
                    return path
        return ""
    except subprocess.CalledProcessError as e:
        print(f"Error downloading video: {e.stderr}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        for name in os.listdir(output_dir):
            if name.startswith("video."):
                try:
                    os.remove(os.path.join(output_dir, name))
                except Exception:
                    pass
        return ""
    except FileNotFoundError:
        print("Error: yt-dlp not found. Please install it: pip install yt-dlp")
        return ""


def extract_frame_at_timestamp(video_path: str, timestamp_seconds: float, output_dir: str, 
                               frame_name: str) -> str:
    """
    Extract a single frame from a video at a specific timestamp using ffmpeg.
    
    Args:
        video_path: Path to the video file
        timestamp_seconds: Timestamp in seconds where to extract the frame
        output_dir: Directory to save extracted frame
        frame_name: Base name for the frame file (without extension)
        
    Returns:
        Path to extracted frame image, or empty string if failed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    frame_filename = f"{frame_name}.jpg"
    frame_path = os.path.join(output_dir, frame_filename)
    
    # Extract frame at specific timestamp
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(timestamp_seconds),
        '-vframes', '1',
        '-q:v', '2',  # High quality
        '-y',  # Overwrite output file
        frame_path
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if os.path.exists(frame_path):
            return frame_path
        else:
            return ""
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not extract frame at {timestamp_seconds}s: {e.stderr}")
        return ""
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return ""


def refine_content_with_openai(title: str, summary: str, client: OpenAI) -> Dict[str, str]:
    """
    Use OpenAI API to refine title and summary into blog-post-friendly HTML format.
    
    Args:
        title: Original title
        summary: Original summary text
        client: OpenAI client instance
        
    Returns:
        Dictionary with 'refined_title' and 'refined_content' (HTML formatted)
    """
    prompt = f"""You are an expert at creating engaging, well-formatted blog posts from lecture content.

Transform the following lecture segment into a polished, blog-post-friendly HTML format:

**Title:** {title}

**Summary:**
{summary}

Please:
1. Create an engaging, clear title (keep it concise but descriptive)
2. Transform the summary into well-structured HTML content with:
   - Proper paragraph breaks
   - Bullet points or numbered lists where appropriate
   - Bold text for key terms and concepts
   - Clear section headers if the content has multiple parts
   - Emphasis on important points
   - Professional, readable formatting

Return your response as a JSON object with two keys:
- "refined_title": The polished title (plain text, no HTML)
- "refined_content": The formatted HTML content (use proper HTML tags like <p>, <ul>, <li>, <strong>, <em>, <h3>, etc.)

Make the content engaging and easy to read while preserving all the important information."""

    try:
        response = client.chat.completions.create(
            model="gpt-api",  # Using gpt-4o-mini (gpt-5-mini doesn't exist yet)
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at formatting educational content into engaging blog posts. Always respond with valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            'refined_title': result.get('refined_title', title),
            'refined_content': result.get('refined_content', f'<p>{summary}</p>')
        }
    except Exception as e:
        print(f"Warning: OpenAI API call failed: {e}")
        print("  Using original content with basic HTML formatting...")
        # Fallback: basic HTML formatting
        formatted_summary = summary.replace('\n\n', '</p><p>').replace('\n', '<br>')
        return {
            'refined_title': title,
            'refined_content': f'<p>{formatted_summary}</p>'
        }


def html_to_plain_text(html: str) -> str:
    """Strip HTML tags and normalize whitespace for chunking-friendly plain text."""
    text = re.sub(r'<[^>]+>', ' ', html)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def write_chunking_files(segments: List[Dict], output_dir: str, video_id: str,
                        metadata: Dict = None) -> None:
    """
    Write chunking-friendly formats for vector DB: one JSON and one JSONL file.
    Each segment is one chunk with plain-text content (no HTML).
    """
    metadata = metadata or {}
    video_url = (metadata.get("video_url") or "").strip()
    chunks = []
    for i, seg in enumerate(segments, 1):
        chunks.append({
            "segment_index": i,
            "video_id": video_id,
            "video_url": video_url,
            "title": seg.get("title", ""),
            "refined_title": seg.get("refined_title", ""),
            "start_timestamp": seg.get("start_timestamp", ""),
            "end_timestamp": seg.get("end_timestamp", ""),
            "content_plain": html_to_plain_text(seg.get("refined_content", "")),
        })
    json_path = os.path.join(output_dir, f"{video_id}_chunks.json")
    jsonl_path = os.path.join(output_dir, f"{video_id}_chunks.jsonl")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "metadata": metadata,
            "segments": chunks
        }, f, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in chunks:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"  Chunking JSON:  {json_path}")
    print(f"  Chunking JSONL: {jsonl_path}")


def _report_title_from_metadata(metadata: Dict) -> str:
    """Build report title from metadata (course_name, number_lecture, year)."""
    parts = []
    course = (metadata.get("course_name") or "").strip()
    lecture = (metadata.get("number_lecture") or "").strip()
    year = (metadata.get("year") or "").strip()
    if course:
        parts.append(course)
    if lecture:
        parts.append(lecture)
    if year:
        parts.append(f"({year})")
    return " – ".join(parts) if parts else "Lecture Summary"


def generate_html_lecture(segments: List[Dict], output_file: str, frames_dir: str = "frames",
                         metadata: Dict = None):
    """
    Generate an HTML lecture summary with embedded images.
    Structure: Title -> Frame -> Content for each segment.

    Args:
        segments: List of segment dictionaries with title, timestamps, summary, refined content, and frame paths
        output_file: Path to output HTML file
        frames_dir: Directory containing the frames (relative path for HTML)
        metadata: Optional dict with course_name, number_lecture, year for the report title
    """
    metadata = metadata or {}
    report_title = _report_title_from_metadata(metadata)
    video_url = (metadata.get("video_url") or "").strip()
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>""" + report_title + """</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 30px;
        }
        .segment {
            margin-bottom: 50px;
            padding: 30px;
            background-color: #fafafa;
            border-left: 4px solid #3498db;
            border-radius: 4px;
        }
        .segment h2 {
            color: #2c3e50;
            margin-top: 0;
            margin-bottom: 20px;
        }
        .segment-frame {
            text-align: center;
            margin: 20px 0;
        }
        .segment-frame img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .segment-content {
            margin-top: 20px;
        }
        .segment-content p {
            margin-bottom: 15px;
        }
        .segment-content ul, .segment-content ol {
            margin-left: 20px;
            margin-bottom: 15px;
        }
        .segment-content li {
            margin-bottom: 8px;
        }
        .segment-content strong {
            color: #2c3e50;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
            font-style: italic;
            margin-bottom: 15px;
        }
        hr {
            border: none;
            border-top: 1px solid #ecf0f1;
            margin: 40px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>""" + report_title + """</h1>
        <p style="color: #7f8c8d; margin-bottom: 30px;">
            This summary highlights the key discussions from the lecture video,
            with visual frames captured at important moments.
        </p>
"""

    if video_url:
        # Display `video_url` in italic for quick reference.
        html_content += f"""
        <p style="color: #7f8c8d; margin-top: -10px; margin-bottom: 30px;">
            Video: <em>{video_url}</em>
        </p>
"""
    
    for i, segment in enumerate(segments, 1):
        refined_title = segment.get('refined_title', segment.get('title', f'Discussion {i}'))
        refined_content = segment.get('refined_content', f'<p>{segment.get("summary", "")}</p>')
        start_ts = segment.get('start_timestamp', 'N/A')
        end_ts = segment.get('end_timestamp', 'N/A')
        frame_paths = segment.get('frame_paths', [])
        
        html_content += f"""
        <div class="segment">
            <h2>{i}. {refined_title}</h2>
            <div class="timestamp">Time: {start_ts} - {end_ts}</div>
"""
        
        # Add frame if available
        if frame_paths:
            frame_path = frame_paths[0]
            # Use relative path for HTML
            rel_path = os.path.relpath(frame_path, os.path.dirname(output_file))
            html_content += f"""
            <div class="segment-frame">
                <img src="{rel_path}" alt="Frame from {start_ts}">
            </div>
"""
        
        # Add refined content
        html_content += f"""
            <div class="segment-content">
                {refined_content}
            </div>
        </div>
"""
        
        if i < len(segments):
            html_content += "<hr>\n"
    
    html_content += """
        <p style="margin-top: 40px; color: #7f8c8d; font-size: 0.9em; text-align: center;">
            Generated automatically from lecture transcript and video analysis.
        </p>
    </div>
</body>
</html>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


def process_video_segments(video_id: str, output_base: str = "reports",
                          skip_download: bool = False):
    """
    Main function to process all video segments.
    
    Args:
        video_id: YouTube video ID (e.g. msHyYioAyNE). URL = https://www.youtube.com/watch?v= + video_id
        json_file: Path to JSON file with segment timestamps/summaries
        output_base: Base output directory; actual output is {output_base}/{video_id}/
        skip_download: If True, skip video download (use existing temp video from a previous run is not used; mainly for re-runs with cached temp)
        openai_api_key: OpenAI API key (if None, uses OPENAI_API_KEY env var)
    """
    youtube_url = YOUTUBE_URL_PREFIX + video_id
    output_dir = os.path.join(output_base, video_id)

    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass --openai-api-key")
        return

    client = OpenAI(api_key=api_key)

    # Create output directories (no videos dir; we do not save the video)
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Load JSON file: expect { "metadata": {...}, "discussions" or "Discussion": [...] }
    json_file_path = os.path.join(output_dir, f"transcript_summary_{video_id}.json")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    metadata = data.get("metadata", {})
    segments = data.get("discussions", data.get("discussion", []))
    if not isinstance(segments, list):
        segments = []

    # Download video to a temporary directory (correct extension for format; avoid "moov atom not found")
    temp_download_dir = tempfile.mkdtemp(prefix="extract_frames_")
    try:
        if not skip_download:
            print("Downloading full video (this may take a while)...")
            full_video_path = download_full_video(youtube_url, temp_download_dir)
            if not full_video_path:
                print("\nError: Failed to download video.")
                print("Try: pip install --upgrade yt-dlp")
                return
            print()
        else:
            full_video_path = ""
            for name in os.listdir(temp_download_dir):
                if name.startswith("video."):
                    full_video_path = os.path.join(temp_download_dir, name)
                    break
            if not full_video_path or not os.path.isfile(full_video_path):
                print("No existing video found; downloading...")
                full_video_path = download_full_video(youtube_url, temp_download_dir)
                if not full_video_path:
                    return
                print()

        print(f"Processing {len(segments)} segments...\n")

        processed_segments = []

        for i, segment in enumerate(segments, 1):
            title = segment.get('title', f'Segment {i}')
            start_ts = segment.get('start_timestamp', '00:00')
            end_ts = segment.get('end_timestamp')
            summary = segment.get('summary', '')

            print(f"[{i}/{len(segments)}] Processing: {title}")
            print(f"  Time: {start_ts} - {end_ts}")

            start_seconds = safe_timestamp_to_seconds(start_ts, 0.0)
            end_seconds = safe_timestamp_to_seconds(end_ts, start_seconds)
            if end_seconds < start_seconds:
                end_seconds = start_seconds

            midpoint_seconds = (end_seconds - start_seconds) / 2 + start_seconds
            midpoint_ts = seconds_to_timestamp(midpoint_seconds)

            print("  Refining content with OpenAI...")
            refined = refine_content_with_openai(title, summary, client)

            print(f"  Extracting frame at midpoint {midpoint_ts}...")
            safe_title = sanitize_filename(title)
            frame_name = f"segment_{i:02d}_{safe_title}"
            frame_path = extract_frame_at_timestamp(full_video_path, midpoint_seconds, frames_dir, frame_name)

            frame_paths = [frame_path] if frame_path else []
            if not frame_path:
                print("  Warning: No frame extracted")

            processed_segments.append({
                'title': title,
                'start_timestamp': start_ts,
                'end_timestamp': end_ts if end_ts else start_ts,
                'refined_title': refined['refined_title'],
                'refined_content': refined['refined_content'],
                'frame_paths': frame_paths
            })
            print()

        # Generate HTML lecture summary (for human reading / lecture summary)
        html_path = os.path.join(output_dir, f"{video_id}_lecture_summary.html")
        
        print(f"Generating HTML lecture summary: {html_path}")
        generate_html_lecture(processed_segments, html_path, frames_dir, metadata=metadata)

        # Write chunking-friendly formats for vector DB (one chunk per segment)
        print("Writing chunking-friendly files (for vector DB):")
        write_chunking_files(processed_segments, output_dir, video_id, metadata=metadata)

        print(f"\nDone! Output saved to: {output_dir}/")
        print(f"  - HTML lecture: {html_path}")
        print(f"  - Chunks (JSON/JSONL): {video_id}_chunks.*")
        print(f"  - Frames: {frames_dir}/")
    finally:
        # Do not keep the downloaded video: remove temp directory and its contents
        if os.path.isdir(temp_download_dir):
            try:
                shutil.rmtree(temp_download_dir, ignore_errors=True)
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from YouTube video segments and generate HTML lecture summary + chunking files'
    )
    parser.add_argument(
        '--video_id',
        default="",
        help='YouTube video ID (e.g. msHyYioAyNE). Full URL: https://www.youtube.com/watch?v=VIDEO_ID'
    )
    parser.add_argument(
        '--output-dir',
        default='../../reports',
        dest='output_base',
        help='Base output directory; output is saved under reports/VIDEO_ID/ (default: reports)'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip video download (video is still not saved; used only when reusing a run)'
    )


    args = parser.parse_args()

    process_video_segments(
        args.video_id,
        args.output_base,
        args.skip_download,
    )


if __name__ == '__main__':
    main()
