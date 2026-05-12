# src/cofit_api_client.py
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Any, Dict, Optional
from src.constants import COFIT_API_URL, COFIT_TOKEN

logger = logging.getLogger(__name__)

RETRY_STRATEGY = Retry(
    total=3,
    backoff_factor=0.5,
    status_forcelist=[408, 429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)


class CofitApiClient:
    def __init__(self, base_url: str = COFIT_API_URL, token: str = COFIT_TOKEN):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        })
        adapter = HTTPAdapter(max_retries=RETRY_STRATEGY)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def get_context_data(
        self,
        skill_key: str,
        client_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """取得 skill config + context data（一次 API call）。

        GET /v5/ai_skills/:skill_key/context_data?client_id=xxx
        回傳: (config, context_data)
        """
        url = f"{self.base_url}/v5/ai_skills/{skill_key}/context_data"
        params: Dict[str, Any] = {"client_id": client_id}
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        response = self.session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        context_data = data.pop("context_data", {})
        return data, context_data
