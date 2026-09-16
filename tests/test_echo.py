from app.processors import EchoProcessor


async def test_echo_returns_input_unchanged() -> None:
    assert await EchoProcessor().process("hello there") == "hello there"
