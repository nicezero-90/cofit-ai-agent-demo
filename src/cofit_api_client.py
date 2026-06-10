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
        """Fetch skill config + context data (single API call).

        GET /v5/ai_skills/:skill_key/context_data?client_id=xxx
        Returns: (config, context_data)
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

    def get_ai_agent_manifest(self, agent_key: str) -> Dict[str, Any]:
        """Fetch agent orchestration structure (without client data).

        GET /v5/ai_agents/:key
        Returns: {key, orchestration_mode, nodes, edges, blocked_nodes, system_prompt, ...}
        """
        url = f"{self.base_url}/v5/ai_agents/{agent_key}"
        response = self.session.get(url, timeout=15)
        response.raise_for_status()
        return response.json()

    def get_ai_agent_context_data(
        self,
        agent_key: str,
        client_id: int,
        skill_keys: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Batch fetch skill config + client data.

        GET /v5/ai_agents/:key/context_data?client_id=X&skill_keys[]=skill_b&...
        Returns: {skills: {skill_key: {...}}, errors: {}}
        """
        url = f"{self.base_url}/v5/ai_agents/{agent_key}/context_data"
        params: Dict[str, Any] = {"client_id": client_id}
        if skill_keys:
            params["skill_keys[]"] = skill_keys
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
