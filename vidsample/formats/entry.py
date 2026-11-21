from dataclasses import asdict, dataclass


@dataclass
class Entry:
    """One member of a remote archive, enough to fetch it later without re-listing."""
    name: str
    size: int            # uncompressed bytes
    offset: int = 0      # format-specific: zip local header / 7z data start / rar header start
    packed: int = 0      # bytes to fetch (compressed size)
    method: int = 0

    def to_dict(self):
        return asdict(self)
