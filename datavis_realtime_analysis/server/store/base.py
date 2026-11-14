"""Backends differ in where bytes live; callers use only these methods. `serve` must 404 on a missing object
(zarrita reads that as an all-fill-value chunk) and support HEAD and Range (sharded arrays)."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import Response


class Store(ABC):
    uri: str
    is_gcs: bool

    @abstractmethod
    def serve(self, path: str, request: Request) -> Response:
        ...

    @abstractmethod
    def list_children(self, rel: str) -> list[str]:
        ...

    @abstractmethod
    def read_attrs(self, rel: str) -> dict | None:
        """None if the node is missing; any other failure raises, so an expired credential never looks like absence."""

    @abstractmethod
    def delete_prefix(self, rel: str) -> int:
        ...

    @abstractmethod
    def replace_group(self, rel: str, write: Callable) -> None:
        """Atomic on local disk; a bucket cannot swap, so readers may briefly see the group empty."""
