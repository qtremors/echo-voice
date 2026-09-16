from app.audio.buffer import AudioBuffer


def test_buffer_caps_data_and_reports_overflow() -> None:
    buffer = AudioBuffer(max_bytes=4)
    assert buffer.append(b"ab") is False
    assert buffer.append(b"cdef") is True
    assert buffer.bytes() == b"abcd"
    buffer.clear()
    assert len(buffer) == 0
