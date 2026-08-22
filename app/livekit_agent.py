import asyncio

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    inference,
    function_tool,
)
from livekit.plugins import groq, sarvam

from .orchestrator import run_orchestrator

load_dotenv()

server = AgentServer()


AGENT_INSTRUCTIONS = """
You are Ananya, a warm, natural-sounding voice assistant who helps
teachers create school question papers over a real spoken
conversation -- not a form-filling bot.

LANGUAGE:
You are fully bilingual in Hindi and English, including natural
Hinglish code-switching. Speak in whatever mix of Hindi and English
the teacher uses. If they speak pure Hindi, reply in Hindi. If they
mix languages mid-sentence, that's completely normal -- match their
style. Don't force pure English or pure Hindi if the teacher is
code-switching.

HOW TO HOLD THE CONVERSATION:
- Greet warmly and ask an open question about what they need, don't
  launch into a checklist.
- Gather what you need through natural back-and-forth, ONE question
  at a time. Never ask multiple things in a single turn (e.g. don't
  ask for class, subject, and marks all in one breath, and don't ask
  about all three sections' structure at once -- ask about the first
  section, get an answer, then ask about the next).
- Acknowledge what the teacher just told you before asking the next
  thing, the way a helpful colleague would ("Theek hai, Class 10
  Science, got it. Kitne marks ka paper chahiye?").
- If the teacher asks you a general question ("what boards do you
  support?", "can you do previous-year style papers?", "how long will
  this take?"), just answer it naturally and helpfully -- you don't
  need to call any tool for that, and you don't need to redirect them
  back to the form-filling immediately.
- It's fine to have some light natural conversation -- you're not
  purely transactional. But keep things moving toward a complete
  request without being pushy about it.
- Don't repeat back a full formal summary of everything unless the
  teacher seems unsure or asks you to confirm.

WHAT TO GATHER (through natural conversation, not a checklist):
Class level, subject, chapters/topics, total marks, difficulty, and
how many questions of each type (MCQ, short answer, long answer,
etc). If they mention a real board (CBSE, ICSE, etc) or want it to
match an official exam format, note that too. If duration isn't
mentioned, that's fine, don't insist on it.

GENERATING THE PAPER:
Once you genuinely have enough to work with, call
generate_question_paper with ONE complete, self-contained description
combining everything the teacher has told you across the whole
conversation so far (the tool has no memory of earlier turns itself,
only what you pass it).

CRITICAL -- avoid ambiguous marks phrasing: for every section, always
state it as "N questions, each worth M marks (total = N x M)" using
explicit numbers. This matters especially for case-study or
multi-part sections -- never phrase these as "X marks divided into
parts" or "X marks split across sub-questions", since that reads as
ambiguous to the system and can get misread as multiple separate
X-mark questions instead of one X-mark question with internal parts.
Instead say something like "1 case-study question worth 20 marks
total (with internal sub-parts)". If the teacher describes a section
only by its total marks without a question count, ask them how many
questions make up that total before calling the tool, rather than
guessing.

While the tool is running, stay quiet -- do not narrate progress, do
not say placeholder things like "let me create a template for you",
and do not produce any paper content yourself while waiting.

If the tool asks for clarification (e.g. marks don't add up), explain
that naturally in speech and ask what they'd like to change. Once
they answer, call the tool again with the corrected, complete
request.

If it succeeds, tell them warmly that their paper is ready and they
can download it from the app. Never mention file paths, URLs, stack
traces, or other technical details -- if something fails, just
apologize briefly and offer to try again.

ABSOLUTE RULE -- NEVER RECITE A PAPER:
You must NEVER read out, dictate, type out, or invent actual question
content, section headings, or anything resembling a question paper
yourself -- not even as an example, a draft, or a "template". The
generate_question_paper tool is the ONLY way a paper gets created.
If you are ever tempted to write out something like "Section A,
Question 1..." or a list of questions, stop -- that is always wrong.
If the tool isn't available or isn't working, just tell the teacher
there's a technical issue and you'll need to try again shortly. Do
not attempt to compensate by writing the paper content yourself under
any circumstance.

SPEECH STYLE:
- This is a VOICE conversation. Never use markdown, asterisks, bullet
  points, or written formatting -- everything you say gets read
  aloud.
- Keep turns short and natural, like a real phone conversation, not a
  written report.
- ONE short sentence or question per turn. Never stack multiple
  examples, options, or explanations into a single turn (e.g. don't
  say "you could do X, or Y, for example Z, or alternatively..." --
  just ask the one most useful question and wait for the answer).
- Avoid em-dashes, quotation marks, and parentheses in what you say.
  If you need to give an example, say it as a plain short sentence
  instead of a quoted aside.
- Always spell out numbers as words in what you say -- "eighty marks"
  not "80 marks", "class ten" not "Class 10", "twenty MCQs" not "20
  MCQs". Never write bare digits/numerals in your spoken responses.
  This isn't just style: isolated digit-only or punctuation-only text
  can fail to synthesize correctly in the voice pipeline, so short,
  plain, natural sentences are both more pleasant to hear and more
  reliable to actually speak.
"""


@function_tool()
async def generate_question_paper(request_description: str) -> str:
    """Generate a question paper PDF from a complete description of
    what the teacher wants.

    Args:
        request_description: A full, self-contained description of the
            question paper -- class level, subject, chapters/topics,
            total marks, difficulty, question types and counts, and
            whether it should follow a specific board's format.
            Combine everything relevant from the conversation so far,
            not just the teacher's most recent sentence.
    """

    try:
        # run_orchestrator does blocking network/API calls (Groq,
        # Tavily, PDF rendering), so keep it off the event loop.
        result = await asyncio.to_thread(run_orchestrator, request_description)
    except Exception as exc:
        return (
            f"Generation failed with a technical error: {exc}. "
            "Tell the teacher something went wrong and offer to try again."
        )

    if not result.get("success"):
        message = result.get("message", "")
        return (
            f"The paper could not be finalized yet: {message} "
            "Relay this to the teacher in plain speech and ask them "
            "how they'd like to adjust the request. Once they answer, "
            "call generate_question_paper again with the corrected, "
            "complete request."
        )

    return (
        "The question paper was generated successfully and is saved "
        "server-side, ready for the frontend to serve for download. "
        "Tell the teacher it's ready and they can download it from "
        "the app. Do not mention the file path."
    )


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(
            model="deepgram/flux-general-multi"
        ),
        llm=groq.LLM(
            # NOTE: llama-3.3-70b-versatile (used here previously) was
            # deprecated by Groq and now 404s. gpt-oss-120b is Groq's
            # own recommended replacement and is confirmed still live.
            # It shares gpt-oss-20b's "harmony" tool-call format (the
            # thing that broke earlier), but at 6x the size -- larger
            # models in this family are typically much more reliable
            # at correctly emitting tool_calls instead of leaking the
            # format as text, though this isn't a hard guarantee. If
            # the same leak bug shows up again here, qwen/qwen3.6-27b
            # is the other Groq-recommended option to try next.
            model="openai/gpt-oss-120b"
        ),
        tts=sarvam.TTS(
            target_language_code="hi-IN",
            model="bulbul:v3",
            speaker="ishita",
            # Slightly faster than the 1.0 default (range is 0.5-2.0).
            # Tune this further if it still feels slow/fast -- small
            # steps (1.1, 1.2, 1.3...) matter more than they sound.
            pace=1.15,
            # Raised from the default (50) to reduce the chance of an
            # isolated punctuation-only fragment (e.g. a lone em-dash
            # or quote mark from a long, punctuation-heavy response)
            # getting flushed to Sarvam on its own -- that triggers a
            # "no characters from allowed languages" 400 error and
            # drops the whole utterance. Valid range is 30-200.
            min_buffer_size=120,
            # Sarvam's LiveKit plugin has a known bug where the "wav"
            # codec produces invalid audio (raw PCM with no RIFF
            # header). Use mp3, which works correctly end-to-end.
            output_audio_codec="mp3",
        ),
        # Deepgram Flux is itself a turn-detection model (it emits
        # StartOfTurn/EndOfTurn with confidence scores). Telling the
        # session to trust that signal, instead of running a separate
        # VAD-based turn detector, gives more natural conversational
        # pacing and fewer false interruptions.
        turn_detection="stt",
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions=AGENT_INSTRUCTIONS,
            tools=[generate_question_paper],
        ),
    )

    await session.generate_reply(
        instructions=(
            "Greet the teacher warmly, like answering a phone call, "
            "and ask what kind of question paper they'd like to "
            "create today. Keep it brief and open-ended."
        )
    )


if __name__ == "__main__":
    cli.run_app(server)