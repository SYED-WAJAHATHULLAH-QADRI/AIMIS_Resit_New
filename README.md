# AI Meeting Intelligence — Streamlit conversion

## Start locally

Use Python 3.11. Install FFmpeg separately and ensure `ffmpeg` is on PATH.

```sh
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

macOS/Linux:
```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Enter your Gemini key in the sidebar, or copy secrets.example.toml into
.streamlit/secrets.toml and replace the placeholder locally. Never commit secrets.
The default model comes from the user's successful Colab experiment; availability
is account-dependent. Generation requests are made only by the Generate button.

## Manual GitHub upload and Streamlit hosting

Extract this ZIP. Upload its code and configuration files to a GitHub repository,
then connect that repository to Streamlit Community Cloud and select app.py as
the entry point. Choose Python 3.11. Enter GEMINI_API_KEY in deployment secrets.
Do not upload your actual secrets file or private meeting evidence. packages.txt
requests FFmpeg on compatible hosted Linux deployments. This package has not
been deployed to your account. Hosting memory/CPU may limit Whisper; start with
tiny.en or base.en. The first transcription downloads model weights; it needs
network access. The Colab GPU is not part of this deployment.

## Workflow

1. Upload one English audio file or UTF-8 transcript, max 50 MB.
2. Transcribe audio locally with Whisper; text inputs bypass Whisper.
3. Check/edit the full transcript, enter reviewer and correction notes, confirm.
4. Generate either/both prompts with the same schema. Every attempt is retained.
5. Download output JSON, Word meeting report, and deterministic email draft.
6. Upload your Colab reference JSON, or download and fill the blank template.
   Supply actual transcript-based decisions/actions with exact quotes and IDs.
   Do not fill a reference from model output and describe it as independent.
7. Match reference tasks to generated items. One combined output item may cover
   several reference tasks. Review fields and support judgments; none default YES.
8. Calculate and download CSV/JSON evaluation and complete session evidence ZIP.

Changing reviewed text requires recording a new review. Old runs stay in evidence
but are excluded from evaluation against the new transcript. Starting another
meeting clears the current session; download first. Closing/refreshing a session
or restarting the host may lose state. No persistent database is included.

## Evaluation and limits

Percentages = 100 * numerator / denominator; zero denominator is null (N/A).
Coverage counts each reference task once, so it accommodates combined outputs.
Field correctness is reported on covered tasks and across all reference tasks.
Unknown-value agreement is included. Additional specified-reference-field
metrics exclude null reference values and penalize missed tasks. Complete task
correctness requires coverage and all three dedicated fields correct.
Support judgments concern extracted decision/task text, not every summary claim.
A matched quotation or schema validation does not establish factual correctness.
These are not conventional one-to-one precision/F1 or whole-output hallucination
rates. No WER, inter-annotator agreement, repeatability statistic, usability rating
or time-saving comparison is fabricated. Multiple runs can be generated and
individually evaluated; aggregate/repeatability analysis is not included.

The original generic and structured prompts are preserved (prompt 1.0, schema
1.1). The app never hardcodes M01 annotations or its reported scores. Audio and
text stay in session memory, except temporary audio files removed after use.
Whisper models are cached globally; transcripts, keys and responses are not.
Gemini receives the reviewed transcript on generation. Email drafts mechanically
mirror the chosen output, including possible errors/omissions; they are not sent
and have not been independently evaluated. Word reports are provided; PDF export
is not included in this version. The pipeline error helper is included and
references are checked against transcript hashes and quotations.

## Tests
```sh
python -m unittest discover -s tests -v
```
Tests cover schema strictness, API schema compatibility, review invalidation,
combined-task evaluation, missing fields, support completeness, downloads and
Streamlit UI loading. Live Gemini and Whisper audio inference require your own
credentials/audio and are not claimed as tested by these offline checks.
