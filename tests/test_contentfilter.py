import os
import pytest
from unittest.mock import patch, MagicMock
from src.contentfilter import run_moderation


@pytest.fixture
def mock_openai_client():
    with patch('src.contentfilter.OpenAI') as MockOpenAI:
        # Mock the moderation result object
        mock_result = MagicMock()
        mock_result.categories.model_dump.return_value = {'hate': False, 'self-harm': False, 'violence': False}
        
        # Mock the response object
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        
        # Mock the OpenAI client
        mock_instance = MockOpenAI.return_value
        mock_instance.moderations.create.return_value = mock_response
        
        yield MockOpenAI


def test_run_moderation(mock_openai_client):
    os.environ['OPENAI_KEY'] = 'fake_key'
    input_text = "This is a sample input"
    expected_output = {'hate': False, 'self-harm': False, 'violence': False}
    
    result = run_moderation(input_text)
    
    assert result == expected_output
    mock_openai_client.assert_called_once_with(api_key='fake_key')
    mock_openai_client.return_value.moderations.create.assert_called_once_with(input=input_text)
