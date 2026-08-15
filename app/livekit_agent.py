from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from dotenv import load_dotenv

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(
            model="deepgram/flux-general-multi"
        )
    )

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="You are a helpful question paper assistant."
        ),
    )


if __name__ == "__main__":
    cli.run_app(server)