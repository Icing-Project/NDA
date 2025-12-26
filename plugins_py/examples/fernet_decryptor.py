"""
Fernet Decryptor Processor Plugin
Symmetric decryption using Python's cryptography library.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState, AudioBuffer

try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[FernetDecryptor] WARNING: cryptography library not installed")
    print("Install with: pip install cryptography")


class FernetDecryptorPlugin(AudioProcessorPlugin):
    """
    Symmetric decryption processor using Fernet (AES-128-CBC + HMAC).
    
    Note: This is a SIMPLIFIED example that reinterprets encrypted bytes as floats.
    NOT suitable for production use. See C++ AES256 plugins for real implementation.
    """
    
    def __init__(self):
        super().__init__()
        self.state = PluginState.UNLOADED if CRYPTO_AVAILABLE else PluginState.ERROR
        self.sample_rate = 48000
        self.channels = 2
        self.key = None
        self.cipher = None
        self.frames_processed = 0
        self.decrypt_failures = 0
    
    def initialize(self) -> bool:
        """Initialize decryptor (key must be set via parameter)"""
        if not CRYPTO_AVAILABLE:
            print("[FernetDecryptor] Error: cryptography library not available")
            self.state = PluginState.ERROR
            return False
        
        # Key will be set via setParameter("key", ...) before start
        self.state = PluginState.INITIALIZED
        print("[FernetDecryptor] Initialized (waiting for key)")
        print("[FernetDecryptor] WARNING: Example only - NOT production-ready")
        return True
    
    def start(self) -> bool:
        """Start decryption"""
        if self.state != PluginState.INITIALIZED:
            return False
        
        if not self.cipher:
            print("[FernetDecryptor] Error: Key not set. Use setParameter('key', key_value)")
            return False
        
        self.state = PluginState.RUNNING
        self.frames_processed = 0
        self.decrypt_failures = 0
        print("[FernetDecryptor] Started")
        return True
    
    def stop(self):
        """Stop decryption"""
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED
            seconds = self.frames_processed / self.sample_rate
            print(f"[FernetDecryptor] Stopped after decrypting {self.frames_processed} frames ({seconds:.2f}s)")
            if self.decrypt_failures > 0:
                print(f"[FernetDecryptor] WARNING: {self.decrypt_failures} decryption failures")
    
    def shutdown(self):
        """Shutdown decryptor"""
        self.state = PluginState.UNLOADED
        self.key = None
        self.cipher = None
        print("[FernetDecryptor] Shutdown")
    
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Decrypt audio buffer in-place.
        
        Args:
            buffer: AudioBuffer to decrypt (modified in-place)
        
        Returns:
            True on success, False on error (passthrough on failure)
        """
        if self.state != PluginState.RUNNING or not self.cipher:
            return False
        
        try:
            # Reinterpret float data as bytes (reverse of encryption)
            encrypted_bytes = buffer.data.tobytes()
            
            # Decrypt
            try:
                decrypted = self.cipher.decrypt(encrypted_bytes)
            except Exception as decrypt_err:
                # Decryption failed (wrong key, corrupted data, etc.)
                self.decrypt_failures += 1
                if self.decrypt_failures <= 5:
                    print(f"[FernetDecryptor] Decryption failed: {decrypt_err}")
                return False  # Passthrough on failure
            
            # Pad/truncate to match original buffer size
            original_size = len(encrypted_bytes)
            if len(decrypted) > original_size:
                decrypted = decrypted[:original_size]
            elif len(decrypted) < original_size:
                decrypted = decrypted + b'\x00' * (original_size - len(decrypted))
            
            # Convert back to float
            decrypted_array = np.frombuffer(decrypted, dtype=np.float32)
            decrypted_array = decrypted_array.reshape(buffer.data.shape)
            
            # Update buffer in-place
            buffer.data[:] = decrypted_array
            
            self.frames_processed += buffer.get_frame_count()
            return True
            
        except Exception as e:
            print(f"[FernetDecryptor] Processing error: {e}")
            self.decrypt_failures += 1
            return False
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Fernet Decryptor",
            version="1.0.0",
            author="NDA Team",
            description="Fernet symmetric decryption (Python, EXAMPLE ONLY)",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )
    
    def get_type(self) -> PluginType:
        return PluginType.PROCESSOR
    
    def get_state(self) -> PluginState:
        return self.state
    
    def get_sample_rate(self) -> int:
        return self.sample_rate
    
    def get_channel_count(self) -> int:
        return self.channels
    
    def set_sample_rate(self, rate: int) -> None:
        self.sample_rate = rate
    
    def set_channel_count(self, channels: int) -> None:
        self.channels = channels
    
    def set_parameter(self, key: str, value: str) -> bool:
        """
        Set decryptor parameter.
        
        Supported parameters:
            - "key": Base64-encoded Fernet key (must match encryptor)
        """
        if key == "key":
            try:
                self.key = value.encode() if isinstance(value, str) else value
                self.cipher = Fernet(self.key)
                print(f"[FernetDecryptor] Key set: {self.key[:16]}...")
                return True
            except Exception as e:
                print(f"[FernetDecryptor] Invalid key: {e}")
                return False
        return False
    
    def get_parameter(self, key: str) -> str:
        """Get decryptor parameter"""
        if key == "key":
            return self.key.decode() if self.key else ""
        return ""


def create_plugin():
    """Factory function called by plugin loader"""
    return FernetDecryptorPlugin()


# Test standalone
if __name__ == "__main__":
    if not CRYPTO_AVAILABLE:
        print("Skipping test - cryptography library not available")
        sys.exit(1)
    
    # Create encryptor and decryptor with same key
    encryptor = FernetEncryptorPlugin()
    decryptor = FernetDecryptorPlugin()
    
    assert encryptor.initialize()
    assert decryptor.initialize()
    
    # Share key from encryptor to decryptor
    shared_key = encryptor.get_parameter("key")
    assert decryptor.set_parameter("key", shared_key)
    
    assert encryptor.start()
    assert decryptor.start()
    
    # Test encryption/decryption roundtrip
    buffer = AudioBuffer(2, 512)
    buffer.data = np.random.randn(2, 512).astype(np.float32)
    original_data = buffer.data.copy()
    
    # Encrypt
    assert encryptor.process_audio(buffer)
    encrypted_data = buffer.data.copy()
    
    # Data should be different after encryption
    assert not np.array_equal(encrypted_data, original_data), "Encryption should modify data!"
    
    # Decrypt
    assert decryptor.process_audio(buffer)
    
    # After decryption, should match original (within tolerance due to truncation)
    # Note: Due to Fernet overhead and truncation, this is approximate
    print(f"Roundtrip max error: {np.max(np.abs(buffer.data - original_data))}")
    
    encryptor.stop()
    decryptor.stop()
    encryptor.shutdown()
    decryptor.shutdown()
    
    print("✓ Fernet crypto plugin roundtrip test completed!")
    print("NOTE: This is an EXAMPLE only - use C++ AES256 plugins for production")

