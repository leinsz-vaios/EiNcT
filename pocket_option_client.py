import time
import requests


class PocketOptionClient:
    def __init__(self, base_url: str, token: str, timeout: int = 10):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
            }
        )

    def get_balance(self, mode: str = 'demo') -> float:
        r = self.session.get(
            f'{self.base_url}/account/balance',
            params={'mode': mode},
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload = r.json()
        return float(payload['balance'])

    def place_order(self, symbol: str, direction: str, amount: float, duration_sec: int, mode: str = 'demo') -> str:
        payload = {
            'symbol': symbol,
            'direction': direction,
            'amount': float(amount),
            'duration_sec': int(duration_sec),
            'mode': mode,
        }
        r = self.session.post(f'{self.base_url}/orders', json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return str(data['order_id'])

    def get_order_result(self, order_id: str):
        r = self.session.get(f'{self.base_url}/orders/{order_id}', timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def wait_for_result(self, order_id: str, poll_interval_sec: float = 1.5, max_wait_sec: int = 180):
        end = time.time() + max_wait_sec
        while time.time() < end:
            data = self.get_order_result(order_id)
            if data.get('status') in ('won', 'lost', 'closed'):
                return data
            time.sleep(poll_interval_sec)
        raise TimeoutError(f'Order {order_id} did not close before timeout.')
