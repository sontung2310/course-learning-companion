from youtube_transcript_api import YouTubeTranscriptApi
from openai import OpenAI
import os
import json
import argparse


def get_client():
    """Initialize OpenAI client (set OPENAI_API_KEY environment variable)."""
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def fetch_transcript_data(video_id: str):
    """
    Fetch raw transcript from YouTube and return list of {text, start, duration}.
    """
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript_data = []
    for snippet in fetched_transcript:
        transcript_data.append({
            "text": snippet.text,
            "start": snippet.start,
            "duration": snippet.duration
        })
    return transcript_data

# Create detailed prompt for summarizing transcript
def create_summarization_prompt(transcript_data):
    """
    Creates a detailed prompt for OpenAI to summarize the transcript
    into key discussions with timestamps.
    """
    # Format transcript with timestamps for context
    transcript_text = ""
    for item in transcript_data:
        minutes = int(item["start"] // 60)
        seconds = int(item["start"] % 60)
        timestamp = f"{minutes:02d}:{seconds:02d}"
        transcript_text += f"[{timestamp}] {item['text']}\n"
    
    prompt = f"""You are an expert at transforming educational video transcripts into clear, easy-to-understand learning materials. Your goal is to create summaries that allow learners to quickly understand lecture content by reading instead of watching videos. Write using simple vocabulary and grammar, and paraphrase complex concepts into simpler terms.

**CRITICAL REQUIREMENTS FOR LEARNING MATERIALS:**
1. **USE SIMPLE LANGUAGE**: Write in clear, simple vocabulary and grammar. Avoid jargon when possible, and when technical terms are necessary, explain them in simple terms. Paraphrase complex ideas into easier-to-understand explanations.
2. **PRESERVE ALL DEFINITIONS**: When a term, concept, or technique is defined, include the complete definition but explain it using simple language. Format clearly as: "[Term]: [Simple, clear definition]"
3. **KEEP IMPORTANT NOTES**: Include warnings, caveats, best practices, common mistakes, and instructor insights that help learners understand the material, written in simple language
4. **EXPLAIN KEY CONCEPTS CLEARLY**: Break down complex concepts into simple explanations. Use everyday language and analogies when helpful
5. **WRITE FOR READABILITY**: Structure the content in a clear, logical flow that is easy to read and understand. Use short sentences, clear headings, bullet points, and paragraphs
6. **PRESERVE CONTEXT**: Include enough context so that definitions and concepts can be understood independently without watching the video

**Instructions:**
1. Analyze the entire transcript and identify the main discussion topics and themes
2. For each key discussion, extract and organize:
   - A clear, descriptive title for the discussion topic
   - ALL definitions mentioned (terms, concepts, techniques, etc.) - explained in simple language
   - Important notes, warnings, or caveats - written simply
   - Key concepts explained clearly and simply
   - The exact timestamp(s) where this discussion occurs (format: MM:SS)
3. Group related discussions together when they span multiple segments
4. Maintain chronological order based on when discussions appear in the video
5. Paraphrase complex technical content into simpler explanations while maintaining accuracy
6. If a discussion spans multiple timestamps, include the start and end times
7. Structure the summary as readable learning material that can replace watching the video

**Output Format:**
Return a JSON object with a single top-level key "discussions". The value of "discussions" must be an array where each element represents one key discussion with the following structure:
{{
    "title": "Brief descriptive title of the discussion",
    "start_timestamp": "MM:SS",
    "end_timestamp": "MM:SS (if discussion spans multiple segments, otherwise null or the same as start_timestamp)",
    "summary": "A clear, easy-to-understand learning material written in simple language. Structure it as follows:\\n\\n{{Summarize content}}:\\n[Main explanation - 2-4 sentences in simple language describing the overall topic and what learners will understand from this section]\\n\\n{{Definitions}}:\\n- [Term 1]: [Simple, clear definition explained in easy-to-understand language]\\n- [Term 2]: [Simple, clear definition explained in easy-to-understand language]\\n\\n{{Key Concepts}}:\\n- [Concept 1 explained simply and clearly, using everyday language when possible]\\n- [Concept 2 explained simply and clearly, using everyday language when possible]\\n\\n{{Important Notes}}:\\n- [Warning, caveat, or important point written in simple language]\\n- [Best practice or common mistake explained simply]"
}}

**Transcript:**
{transcript_text}

**Now analyze this transcript and provide the key discussions in the specified JSON format. Always return a single JSON object with a top-level \"discussions\" array:**"""

    return prompt

# Generate summary using OpenAI
def summarize_transcript(transcript_data, client=None):
    """
    Uses OpenAI API to summarize the transcript into key discussions.
    """
    if client is None:
        client = get_client()
    prompt = create_summarization_prompt(transcript_data)
    
    response = client.chat.completions.create(
        model="gpt-api",  # Using gpt-4o-mini (gpt-5-mini doesn't exist yet)
        messages=[
            {
                "role": "system",
                "content": "You are an expert at transforming educational video transcripts into clear, easy-to-understand learning materials. Your summaries must use simple vocabulary and grammar, paraphrase complex concepts into simpler terms, and preserve all definitions and important notes. Write in a clear, simple style that allows learners to understand the lecture content easily by reading instead of watching. Always respond with valid JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        # temperature=0.3,  # Lower temperature for more consistent, factual output
        response_format={"type": "json_object"}  # Ensure JSON output
    )
    
    return response.choices[0].message.content


def run_pipeline(
    video_id: str,
    output_dir: str = ".",
    verbose: bool = True,
    course_name: str = "",
    number_lecture: str = "",
    # year: str = "",
):
    """
    Full pipeline: fetch transcript -> summarize -> add metadata -> save to transcript_summary_{video_id}.json.
    Returns path to the saved JSON file.
    Metadata (video_id, course_name, number_lecture) is included for RAG use.
    """
    if verbose:
        print("Fetching transcript...")
    transcript_data = fetch_transcript_data(video_id)
    if verbose:
        print(f"Transcript length: {len(transcript_data)} snippets\n")
        print("Generating summary with OpenAI API...")
    client = get_client()
    summary_json = summarize_transcript(transcript_data, client=client)
    summary_data = json.loads(summary_json)
    if isinstance(summary_data, dict) and "discussions" in summary_data:
        discussions = summary_data["discussions"]
    elif isinstance(summary_data, list):
        discussions = summary_data
    else:
        raise ValueError(f"Unexpected summary_data structure: {type(summary_data)}")

    # Add metadata for RAG
    metadata = {
        "video_id": video_id,
        "course_name": course_name,
        "number_lecture": number_lecture,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        # "year": year,
    }
    output = {"metadata": metadata, "discussions": discussions}

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, video_id), exist_ok=True)
    out_path = os.path.join(output_dir, video_id, f"transcript_summary_{video_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"\nSummary saved to '{out_path}'")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch YouTube transcript and summarize with OpenAI")
    parser.add_argument("--video_id", default="", help="YouTube video ID (e.g. ptFiH_bHnJw)")
    parser.add_argument("--output-dir", default="../../reports", help="Directory for transcript_summary_{video_id}.json")
    parser.add_argument("--course_name", default="", help="Course name (stored in metadata for RAG)")
    parser.add_argument(
        "--number_lecture",
        default="",
        help='Lecture identifier (e.g. "Lecture 2: Pytorch, Resource Accounting")',
    )
    # parser.add_argument("--year", default="2025", help="Year of the course")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()
    run_pipeline(
        video_id=args.video_id,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        course_name=args.course_name,
        number_lecture=args.number_lecture,
        # year=args.year,
    )
