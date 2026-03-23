"""
Airflow DAG: Lecture pipeline
Trigger with a JSON config containing three keys:
  {
    "video_id":       "6OBtO9niT00",
    "course_name":    "Stanford CS336 Language Modeling from Scratch",
    "number_lecture": "Lecture 5: GPUs"
  }

Pipeline steps:
  1. fetch_transcript  → transcript_API        → reports/{video_id}/transcript_summary_{video_id}.json
  2. refine_report     → refine_report          → reports/{video_id}/{video_id}_lecture_summary.html
                                                   reports/{video_id}/{video_id}_chunks.json/jsonl
  3. ingest_chromadb  → ingest_chromadb         → ChromaDB upsert of chunked segments
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# Add dags directory to sys.path so sibling modules are importable
_DAGS_DIR = os.path.realpath(os.path.dirname(__file__))
if _DAGS_DIR not in sys.path:
    sys.path.insert(0, _DAGS_DIR)

def _default_pipeline_output_base() -> str:
    """
    Local dev: repo `reports/` next to `ingest_data/`.
    Docker (official compose): only `.../dags` is mounted at /opt/airflow/dags, so
    dirname/../.. is /opt, not the repo — writing /opt/reports fails with PermissionError.
    Use Airflow's data dir when present, or set LECTURE_PIPELINE_OUTPUT_BASE explicitly.
    """
    if os.path.isdir("/opt/airflow/reports"):
        return "/opt/airflow/reports"
    if os.path.isdir("/opt/airflow/data"):
        return os.path.join("/opt/airflow/data", "reports")
    return os.path.realpath(os.path.join(_DAGS_DIR, "..", "..", "reports"))


# Base output directory for all pipeline reports (one sub-folder per video_id)
PIPELINE_OUTPUT_BASE = os.environ.get(
    "LECTURE_PIPELINE_OUTPUT_BASE",
    _default_pipeline_output_base(),
)

DEFAULT_VIDEO_ID = "ptFiH_bHnJw"


def _get_conf(**context):
    """Return (video_id, course_name, number_lecture) from DAG run config or Airflow Variables."""
    dag_run = context.get("dag_run")
    conf = (getattr(dag_run, "conf", None) or {}) if dag_run else {}
    video_id = conf.get("video_id") or Variable.get(
        "lecture_pipeline_video_id", default_var=DEFAULT_VIDEO_ID
    )
    course_name = conf.get("course_name") or Variable.get(
        "lecture_pipeline_course_name", default_var=""
    )
    number_lecture = conf.get("number_lecture") or Variable.get(
        "lecture_pipeline_number_lecture", default_var=""
    )
    return video_id, course_name, number_lecture


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------

def task_fetch_transcript(**context):
    """
    Step 1 — transcript_API:
      Fetch YouTube transcript, summarise with OpenAI, and save
      reports/{video_id}/transcript_summary_{video_id}.json.
    Equivalent CLI:
      python transcript_API.py --video_id 6OBtO9niT00
        --output-dir "../../reports"
        --course_name "Stanford CS336 Language Modeling from Scratch"
        --number_lecture "Lecture 5: GPUs"
    """
    video_id, course_name, number_lecture = _get_conf(**context)
    import transcript_API  # noqa: PLC0415

    out_path = transcript_API.run_pipeline(
        video_id=video_id,
        output_dir=PIPELINE_OUTPUT_BASE,
        verbose=True,
        course_name=course_name,
        number_lecture=number_lecture,
    )
    ti = context["ti"]
    ti.xcom_push(key="video_id", value=video_id)
    ti.xcom_push(key="transcript_summary_path", value=out_path)
    return out_path


def task_refine_report(**context):
    """
    Step 2 — refine_report:
      Download the YouTube video, extract frames at each segment midpoint,
      refine content with OpenAI, and write:
        - reports/{video_id}/{video_id}_lecture_summary.html
        - reports/{video_id}/{video_id}_chunks.json / .jsonl
    Equivalent CLI:
      python refine_report.py --video_id "6OBtO9niT00"
    """
    ti = context["ti"]
    video_id = ti.xcom_pull(task_ids="fetch_transcript", key="video_id")
    transcript_path = ti.xcom_pull(task_ids="fetch_transcript", key="transcript_summary_path")
    if not transcript_path or not os.path.isfile(transcript_path):
        raise FileNotFoundError(f"Transcript summary not found: {transcript_path}")

    import refine_report  # noqa: PLC0415

    refine_report.process_video_segments(
        video_id=video_id,
        output_base=PIPELINE_OUTPUT_BASE,
    )
    html_path = os.path.join(PIPELINE_OUTPUT_BASE, video_id, f"{video_id}_lecture_summary.html")
    ti.xcom_push(key="html_path", value=html_path)
    return html_path


def task_ingest_chromadb(**context):
    """
    Step 3 — ingest_chromadb:
      Embed chunked segments with SentenceTransformer and upsert into ChromaDB.
    Equivalent CLI:
      python ingest_chromadb.py --video_id "6OBtO9niT00"
    """
    ti = context["ti"]
    video_id = ti.xcom_pull(task_ids="fetch_transcript", key="video_id")
    if not video_id:
        raise ValueError("Missing video_id from upstream task")

    import ingest_chromadb  # noqa: PLC0415

    n = ingest_chromadb.ingest_to_chromadb(
        input_path=PIPELINE_OUTPUT_BASE,
        video_id=video_id,
        collection_name=ingest_chromadb.DEFAULT_COLLECTION_NAME,
        persist_directory=os.environ.get("CHROMA_PERSIST_DIR", ingest_chromadb.DEFAULT_PERSIST_DIR),
    )
    print(f"Ingested {n} chunks for video_id={video_id}")
    return n


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id="lecture_pipeline",
    default_args={
        "owner": "airflow"
    },
    description=(
        "Lecture pipeline: fetch transcript → refine report (frames + HTML) → ingest ChromaDB"
    ),
    schedule_interval=None,  # manual / API trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["lecture", "youtube", "chromadb"],
    # Default params shown in the Airflow UI "Trigger DAG w/ config" form
    params={
        "video_id": DEFAULT_VIDEO_ID,
        "course_name": "",
        "number_lecture": "",
    },
) as dag:

    fetch_transcript = PythonOperator(
        task_id="fetch_transcript",
        python_callable=task_fetch_transcript,
    )

    refine_report_task = PythonOperator(
        task_id="refine_report",
        python_callable=task_refine_report,
    )

    ingest_chromadb_task = PythonOperator(
        task_id="ingest_chromadb",
        python_callable=task_ingest_chromadb,
    )

    fetch_transcript >> refine_report_task >> ingest_chromadb_task
