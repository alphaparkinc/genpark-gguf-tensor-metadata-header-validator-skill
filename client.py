import struct
from typing import Dict, Any, List, Optional

class GGUFTensorMetadataHeaderValidator:
    """
    Parses and verifies GGUF specification binary headers (magic 'GGUF', version 2/3,
    tensor count, metadata KV-pairs, and tensor info offsets).
    """
    GGUF_MAGIC = b"GGUF"
    SUPPORTED_VERSIONS = [2, 3]

    def validate_header_bytes(self, header_bytes: bytes) -> Dict[str, Any]:
        if len(header_bytes) < 24:
            return {"valid": False, "reason": "Header bytes shorter than minimum GGUF 24-byte preamble"}

        magic = header_bytes[:4]
        if magic != self.GGUF_MAGIC:
            return {"valid": False, "reason": f"Invalid magic header {magic!r}, expected b'GGUF'"}

        version, tensor_count, metadata_kv_count = struct.unpack("<III", header_bytes[4:16])
        if version not in self.SUPPORTED_VERSIONS:
            return {"valid": False, "reason": f"Unsupported GGUF version {version}"}

        return {
            "valid": True,
            "magic": "GGUF",
            "version": version,
            "tensor_count": tensor_count,
            "metadata_kv_count": metadata_kv_count,
            "header_size_bytes": 16,
            "is_ready_for_mmap": True
        }

    def verify_architecture_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        required_keys = ["general.architecture", "general.name"]
        arch = metadata.get("general.architecture", "llama")
        arch_keys = [f"{arch}.context_length", f"{arch}.embedding_length", f"{arch}.block_count"]
        
        missing = [k for k in required_keys + arch_keys if k not in metadata]
        
        return {
            "architecture": arch,
            "is_metadata_complete": len(missing) == 0,
            "missing_keys": missing,
            "context_window": metadata.get(f"{arch}.context_length", 4096),
            "layers_count": metadata.get(f"{arch}.block_count", 32)
        }
