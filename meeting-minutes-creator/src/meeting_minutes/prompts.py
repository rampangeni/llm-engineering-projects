"""Prompts for the meeting-minutes generation."""

SYSTEM_MESSAGE = (
    "You produce minutes of meetings from transcripts, with summary, key discussion points, "
    "takeaways and action items with owners, in markdown."
)


def user_prompt(transcription: str, meeting_description: str = "a Denver City council meeting") -> str:
    return (
        f"Below is an extract transcript of {meeting_description}. "
        "Please write minutes in markdown, including a list of attendees (if present), summary, "
        "key discussion points, takeaways and action items with owners.\n"
        f"{transcription}"
    )


def build_messages(transcription: str, meeting_description: str = "a Denver City council meeting") -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_prompt(transcription, meeting_description)},
    ]
