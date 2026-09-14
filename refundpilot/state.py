"""SQLite-backed business state, isolated once per harness session."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import TaskCase


SCHEMA = """
CREATE TABLE metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    item TEXT NOT NULL,
    amount REAL NOT NULL,
    status TEXT NOT NULL,
    delivered_at TEXT,
    refunded INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE refunds (
    refund_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE escalations (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

DROP_SCHEMA = """
DROP TABLE IF EXISTS escalations;
DROP TABLE IF EXISTS refunds;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS metadata;
"""


class WorldState:
    def __init__(self, path: Path, case: TaskCase):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)
        self._seed(case)

    def reset(self, case: TaskCase) -> None:
        """Restore this isolated database to the task's initial state."""

        self.connection.executescript(DROP_SCHEMA)
        self.connection.executescript(SCHEMA)
        self._seed(case)

    def _seed(self, case: TaskCase) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("as_of", json.dumps(case.as_of)),
            )
            self.connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("policy", json.dumps(case.policy, ensure_ascii=False)),
            )
            for order in case.orders:
                self.connection.execute(
                    """
                    INSERT INTO orders(
                        order_id, customer_id, item, amount, status,
                        delivered_at, refunded
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        order["order_id"],
                        order.get("customer_id", "unknown"),
                        order.get("item", "unknown"),
                        float(order.get("amount", 0.0)),
                        order.get("status", "unknown"),
                        order.get("delivered_at"),
                        int(bool(order.get("refunded", False))),
                    ),
                )
            for refund in case.initial_refunds:
                self.connection.execute(
                    "INSERT INTO refunds(order_id, reason, created_at) VALUES (?, ?, ?)",
                    (
                        refund["order_id"],
                        refund.get("reason", "existing refund"),
                        refund.get("created_at", case.as_of),
                    ),
                )

    def metadata(self, key: str) -> Any:
        row = self.connection.execute(
            "SELECT value FROM metadata WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        return json.loads(row["value"])

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["refunded"] = bool(value["refunded"])
        return value

    def has_refund(self, order_id: str) -> bool:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM refunds WHERE order_id = ?", (order_id,)
        ).fetchone()
        return bool(row["count"])

    def create_refund(self, order_id: str, reason: str) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO refunds(order_id, reason, created_at) VALUES (?, ?, ?)",
                (order_id, reason, self.metadata("as_of")),
            )
            self.connection.execute(
                "UPDATE orders SET refunded = 1 WHERE order_id = ?", (order_id,)
            )
        return int(cursor.lastrowid)

    def create_escalation(self, order_id: str, reason: str) -> int:
        existing = self.connection.execute(
            "SELECT escalation_id FROM escalations WHERE order_id = ?", (order_id,)
        ).fetchone()
        if existing is not None:
            return int(existing["escalation_id"])
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO escalations(order_id, reason, created_at) VALUES (?, ?, ?)",
                (order_id, reason, self.metadata("as_of")),
            )
        return int(cursor.lastrowid)

    def summary(self) -> Dict[str, Any]:
        rows = self.connection.execute(
            "SELECT * FROM orders ORDER BY order_id"
        ).fetchall()
        orders: List[Dict[str, Any]] = []
        for row in rows:
            order = dict(row)
            order["refunded"] = bool(order["refunded"])
            orders.append(order)
        refund_count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM refunds"
        ).fetchone()["count"]
        escalation_count = self.connection.execute(
            "SELECT COUNT(*) AS count FROM escalations"
        ).fetchone()["count"]
        return {
            "orders": orders,
            "refunded_order_ids": sorted(
                order["order_id"] for order in orders if order["refunded"]
            ),
            "refund_count": int(refund_count),
            "escalation_count": int(escalation_count),
        }

    def close(self) -> None:
        self.connection.close()
