from email.message import Message
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from qev.client import Client, Noul


PAYLOAD = {"model": "test", "answers": {"ok": {"type": "noul", "noul": 0.5}}}


def test_query_values_are_url_encoded():
    client = Client()
    with patch.object(client, "_post", return_value=(PAYLOAD, {})) as post:
        client.system_one(None, {"ok": Noul()}, note="a &b=2/#中文", debug=False)
    query = parse_qs(urlsplit(post.call_args.args[0]).query)
    assert query == {"note": ["a &b=2/#中文"], "debug": ["false"]}


def test_request_id_header_is_case_insensitive():
    client = Client()
    headers = Message()
    headers["X-Typesafe-Request-Id"] = "request-123"
    with patch.object(client, "_post", return_value=(PAYLOAD, dict(headers))):
        response = client.system_one(None, {"ok": Noul()})
    assert response.request_id == "request-123"
