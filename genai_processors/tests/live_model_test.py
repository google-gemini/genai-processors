import unittest
from unittest import mock

from absl.testing import absltest
from genai_processors import content_api
from genai_processors import streams
from genai_processors.core import live_model
from google.genai import live
from google.genai import types as genai_types


def mock_live_connect(return_value):
  client_connection = mock.AsyncMock(spec=live.AsyncSession)
  client_connection.send_client_content = mock.AsyncMock()
  client_connection.send_realtime_input = mock.AsyncMock()
  client_connection.send_tool_response = mock.AsyncMock()
  client_connection.close = mock.AsyncMock()

  async def receive_mock():
    if return_value:
      yield return_value

  client_connection.receive.return_value = receive_mock()

  return client_connection


class SimpleManagerMock:

  def __init__(self, client_connection):
    self._client_connection = client_connection

  async def __aenter__(self):
    return self._client_connection

  async def __aexit__(self, exc_type, exc_value, traceback):
    return None


class LiveModelTest(unittest.IsolatedAsyncioTestCase):

  async def test_connect_and_send_content(self):
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(turn_complete=True),
        )
    )

    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = [
          content_api.ProcessorPart('test_content_1', role='user'),
          content_api.ProcessorPart(
              'test_content_2',
              role='model',
              metadata={'turn_complete': False},
          ),
      ]
      await streams.gather_stream(
          model(streams.stream_content(content, with_delay_sec=0.1))
      )

      client_connection.send_client_content.assert_has_calls([
          mock.call(
              turns=genai_types.Content(parts=[content[0]], role='user'),
              turn_complete=True,
          ),
          mock.call(
              turns=genai_types.Content(parts=[content[1]], role='model'),
              turn_complete=False,
          ),
      ])

  async def test_connect_and_send_realtime_media(self):
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(turn_complete=True),
        )
    )

    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )

      # substream name is not set. Test that nothing is sent to the model.
      content_bytes = b'test_image_bytes'
      content = [
          content_api.ProcessorPart(content_bytes, mimetype='image/png'),
      ]
      async for _ in model(streams.stream_content(content)):
        pass

      client_connection.send_realtime_input.assert_not_called()

      # substream name is set to 'realtime'. Test that the image is sent to the
      # model.
      content = [
          content_api.ProcessorPart(
              content_bytes, mimetype='image/png', substream_name='realtime'
          ),
      ]
      async for _ in model(streams.stream_content(content, with_delay_sec=0.1)):
        pass
      client_connection.send_realtime_input.assert_called_once_with(
          media=genai_types.Blob(data=content_bytes, mime_type='image/png')
      )

  async def test_receive_audio_with_transcription(self):
    # Test all fields of the server message.
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            server_content=genai_types.LiveServerContent(
                model_turn=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            inline_data=genai_types.Blob(
                                data=b'test_content',
                                mime_type='audio/wav',
                            ),
                        ),
                    ],
                    role='model',
                ),
                turn_complete=True,
                generation_complete=True,
                interrupted=True,
                output_transcription=genai_types.Transcription(
                    text='transcription_out',
                ),
                input_transcription=genai_types.Transcription(
                    text='transcription_in',
                ),
            ),
        )
    )

    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = content_api.ProcessorPart('test_content_1')
      output_content = []
      async for part in model(
          streams.stream_content([content], with_delay_sec=0.1)
      ):
        output_content.append(part)

      expected = [
          content_api.ProcessorPart(
              value=b'test_content',
              role='model',
              mimetype='audio/wav',
          ),
          content_api.ProcessorPart(
              value='',
              role='model',
              metadata={
                  'turn_complete': True,
              },
          ),
          content_api.ProcessorPart(
              value='',
              role='model',
              metadata={
                  'interrupted': True,
              },
          ),
          content_api.ProcessorPart(
              value='',
              role='model',
              metadata={
                  'generation_complete': True,
              },
          ),
          content_api.ProcessorPart(
              value='transcription_in',
              role='model',
              substream_name='input_transcription',
          ),
          content_api.ProcessorPart(
              value='transcription_out',
              role='model',
              substream_name='output_transcription',
          ),
      ]
      self.assertEqual(output_content, expected)

  async def test_receive_tool_call(self):
    # Test all fields of the server message.
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            tool_call=genai_types.LiveServerToolCall(
                function_calls=[
                    genai_types.FunctionCall(
                        name='fn_1',
                        args={'x': '1', 'y': 2},
                        id='1',
                    ),
                    genai_types.FunctionCall(
                        name='fn_2',
                        args={'x': '2', 'y': 2},
                        id='2',
                    ),
                ],
            )
        )
    )

    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = content_api.ProcessorPart('test_content_1')
      output_content = []
      async for part in model(
          streams.stream_content([content], with_delay_sec=0.1)
      ):
        output_content.append(part)

      expected = [
          content_api.ProcessorPart(
              value=genai_types.Part.from_function_call(
                  name='fn_1', args={'x': '1', 'y': 2}
              ),
              metadata={'id': '1'},
              role='model',
          ),
          content_api.ProcessorPart(
              value=genai_types.Part.from_function_call(
                  name='fn_2', args={'x': '2', 'y': 2}
              ),
              metadata={'id': '2'},
              role='model',
          ),
      ]
      self.assertEqual(output_content, expected)

  async def test_receive_tool_call_cancellation(self):
    # Test all fields of the server message.
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            tool_call_cancellation=genai_types.LiveServerToolCallCancellation(
                ids=['1', '2'],
            )
        )
    )
    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = content_api.ProcessorPart('test_content_1')
      output_content = []
      async for part in model(
          streams.stream_content([content], with_delay_sec=0.1)
      ):
        output_content.append(part)
      expected = [
          content_api.ProcessorPart.from_tool_cancellation(
              function_call_id='1'
          ),
          content_api.ProcessorPart.from_tool_cancellation(
              function_call_id='2'
          ),
      ]
      self.assertEqual(output_content, expected)

  async def test_receive_usage_metadata(self):
    # Test all fields of the server message.
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            usage_metadata=genai_types.UsageMetadata(total_token_count=10)
        )
    )
    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = content_api.ProcessorPart('test_content_1')
      output_content = []
      async for part in model(
          streams.stream_content([content], with_delay_sec=0.1)
      ):
        output_content.append(part)
      expected = [
          content_api.ProcessorPart(
              value='',
              role='model',
              metadata={'usage_metadata': {'total_token_count': 10}},
          ),
      ]
      self.assertEqual(output_content, expected)

  async def test_receive_session_resumption_update(self):
    # Test all fields of the server message.
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            session_resumption_update=genai_types.LiveServerSessionResumptionUpdate(
                new_handle='test_handle',
                resumable=True,
                last_consumed_client_message_index=1,
            ),
        )
    )
    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = content_api.ProcessorPart('test_content_1')
      output_content = []
      async for part in model(
          streams.stream_content([content], with_delay_sec=0.1)
      ):
        output_content.append(part)
      expected = [
          content_api.ProcessorPart(
              value='',
              role='model',
              metadata={
                  'session_resumption_update': {
                      'new_handle': 'test_handle',
                      'resumable': True,
                      'last_consumed_client_message_index': 1,
                  }
              },
          ),
      ]
      self.assertEqual(output_content, expected)

  async def test_receive_go_away(self):
    # Test all fields of the server message.
    client_connection = mock_live_connect(
        return_value=genai_types.LiveServerMessage(
            go_away=genai_types.LiveServerGoAway(time_left='10')
        )
    )
    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      content = content_api.ProcessorPart('test_content_1')
      output_content = []
      async for part in model(
          streams.stream_content([content], with_delay_sec=0.1)
      ):
        output_content.append(part)
      expected = [
          content_api.ProcessorPart(
              value='',
              role='model',
              metadata={'go_away': {'time_left': '10'}},
          ),
      ]
      self.assertEqual(output_content, expected)

  async def test_raise_exception(self):
    client_connection = mock_live_connect('')
    client_connection.receive.side_effect = IOError('test exception')
    with mock.patch.object(
        live.AsyncLive,
        'connect',
        return_value=SimpleManagerMock(client_connection),
    ):
      model = live_model.LiveProcessor(
          api_key='test_api_key',
          model_name='test_model_name',
      )
      with self.assertRaises(IOError):
        await streams.gather_stream(
            model(
                streams.stream_content(
                    [content_api.ProcessorPart('test_content_1')]
                )
            )
        )


if __name__ == '__main__':
  absltest.main()
