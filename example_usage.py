import json, struct
from client import GGUFTensorMetadataHeaderValidator

def main():
    validator = GGUFTensorMetadataHeaderValidator()
    # Synthesize valid GGUF binary header: b'GGUF' + version 3 + 291 tensors + 24 metadata keys
    raw_header = b"GGUF" + struct.pack("<III", 3, 291, 24) + b"\x00" * 8
    res = validator.validate_header_bytes(raw_header)
    print("GGUF Header Validation:")
    print(json.dumps(res, indent=2))
    assert res["valid"] is True
    assert res["version"] == 3
    assert res["tensor_count"] == 291
    
    meta = {
        "general.architecture": "llama",
        "general.name": "Llama-3.1-8B-Instruct",
        "llama.context_length": 131072,
        "llama.embedding_length": 4096,
        "llama.block_count": 32
    }
    arch_res = validator.verify_architecture_metadata(meta)
    assert arch_res["is_metadata_complete"] is True
    print("GGUF validator verification: PASS")

if __name__ == "__main__":
    main()
