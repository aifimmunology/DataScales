from contextlib import contextmanager

from google.api_core.exceptions import NotFound


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    @property
    def size(self):
        return len(self.bucket.objects[self.name])

    def download_as_bytes(self, start=None, end=None):
        if self.name not in self.bucket.objects:
            raise NotFound(self.name)
        data = self.bucket.objects[self.name]
        if start is None:
            return data
        return data[start:] if end is None and start >= 0 else data[start:(end + 1) if end is not None else None]

    def upload_from_filename(self, path, content_type=None):
        self.bucket.objects[self.name] = open(path, "rb").read()

    def delete(self):
        self.bucket.objects.pop(self.name, None)


class _Listing:
    def __init__(self, blobs, prefixes):
        self.blobs, self.prefixes = blobs, prefixes
        self.pages = [type("Page", (), {"__iter__": lambda s: iter(()), "prefixes": prefixes})()]

    def __iter__(self):
        return iter(self.blobs)


class FakeBucket:
    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        return FakeBlob(self, name) if name in self.objects else None

    def list_blobs(self, prefix="", delimiter=None):
        names = sorted(n for n in self.objects if n.startswith(prefix))
        if delimiter:
            kids = {prefix + n[len(prefix):].split(delimiter, 1)[0] + delimiter for n in names if delimiter in n[len(prefix):]}
            return _Listing([], sorted(kids))
        return [FakeBlob(self, n) for n in names]


class FakeGcs:
    def __init__(self):
        self.buckets: dict[str, FakeBucket] = {}

    def bucket(self, name):
        return self.buckets.setdefault(name, FakeBucket())

    @contextmanager
    def batch(self, raise_exception=True):
        yield
