import json
from pathlib import Path

import httpx
import pytest

from facade_mcp.prompt_selection import judge, selector_config


@pytest.mark.parametrize("url", ["http://example.com", "https://key@example.com", "https://example.com?key=secret"])
def test_selector_rejects_insecure_configuration(monkeypatch, url):
    monkeypatch.setenv("FACADE_SELECTOR_URL", url)
    monkeypatch.setenv("FACADE_SELECTOR_MODEL", "judge")
    monkeypatch.setenv("FACADE_SELECTOR_API_KEY", "secret")
    with pytest.raises(ValueError):
        selector_config()


@pytest.mark.parametrize("body,valid", [('1', False), ('{"selected_mode":"other","reason":"x"}', False),
    ('{"selected_mode":"text","reason":"clear boundary"}', True)])
def test_judge_strict_response_and_no_retry(tmp_path, monkeypatch, body, valid):
    from facade_mcp import prompt_selection
    image = tmp_path / "overlay.png"
    image.write_bytes(b"fixture only")
    class Store:
        def resolve_input(self, value):
            return image
    calls = []
    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": body}}]})
    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))
    candidates = {mode: {"overlay": {"uri": "artifact:fixture"}} for mode in ("bbox", "point", "text")}
    if valid:
        result = judge(Store(), candidates, ("https://example.invalid/chat/completions", "judge", "secret"))
        assert result["selected_mode"] == "text" and not result["is_ground_truth"]
    else:
        with pytest.raises(ValueError, match="no fallback"):
            judge(Store(), candidates, ("https://example.invalid/chat/completions", "judge", "secret"))
    assert len(calls) == 1
    assert len([item for item in calls[0]["messages"][0]["content"] if item["type"] == "image_url"]) == 3
