"""
Fernet Encryptor Processor Plugin
Symmetric encryption using Python's cryptography library.
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
    print("[FernetEncryptor] WARNING: cryptography library not installed")
    print("Install with: pip install cryptography")


class FernetEncryptorPlugin(AudioProcessorPlugin):
    """
    Symmetric encryption processor using Fernet (AES-128-CBC + HMAC).
    
    Note: Fernet adds overhead (base64 encoding, timestamp, HMAC), so encrypted
    data is larger than plaintext. This plugin truncates to fit buffer size,
    which is NOT suitable for production. Use for testing only.
    
    For production, use streaming cipher modes (e.g., AES-GCM in C++ plugin).
    """
    
    def __init__(self):
        super().__init__()
        self.state = PluginState.UNLOADED if CRYPTO_AVAILABLE else PluginState.ERROR
        self.sample_rate = 48000
        self.channels = 2
        self.key = None
        self.cipher = None
        self.frames_processed = 0
    
    def initialize(self) -> bool:
        """Initialize encryptor with random key"""
        if not CRYPTO_AVAILABLE:
            print("[FernetEncryptor] Error: cryptography library not available")
            self.state = PluginState.ERROR
            return False
        
        try:
            # Generate random key (in production, load from secure storage)
            self.key = Fernet.generate_key()
            self.cipher = Fernet(self.key)
            self.state = PluginState.INITIALIZED
            print(f"[FernetEncryptor] Initialized with key: {self.key[:16]}...")
            print("[FernetEncryptor] WARNING: Example only - NOT production-ready")
            return True
        except Exception as e:
            print(f"[FernetEncryptor] Initialization failed: {e}")
            self.state = PluginState.ERROR
            return False
    
    def start(self) -> bool:
        """Start encryption"""
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        self.frames_processed = 0
        print("[FernetEncryptor] Started")
        return True
    
    def stop(self):
        """Stop encryption"""
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED
            seconds = self.frames_processed / self.sample_rate
            print(f"[FernetEncryptor] Stopped after encrypting {self.frames_processed} frames ({seconds:.2f}s)")
    
    def shutdown(self):
        """Shutdown encryptor"""
        self.state = PluginState.UNLOADED
        self.key = None
        self.cipher = None
        print("[FernetEncryptor] Shutdown")
    
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Encrypt audio buffer in-place.
        
        Args:
            buffer: AudioBuffer to encrypt (modified in-place)
        
        Returns:
            True on success, False on error
        """
        if self.state != PluginState.RUNNING or not self.cipher:
            return False
        
        try:
            # Convert float audio to bytes
            audio_bytes = buffer.data.tobytes()
            
            # Encrypt (adds overhead: IV, timestamp, HMAC)
            encrypted = self.cipher.encrypt(audio_bytes)
            
            # WARNING: This truncates to fit buffer - NOT suitable for production!
            # In production, use streaming cipher or handle size increase properly
            original_size = len(audio_bytes)
            if len(encrypted) > original_size:
                encrypted = encrypted[:original_size]
                # Zero-pad if shorter
            elif len(encrypted) < original_size:
                encrypted = encrypted + b'\x00' * (original_size - len(encrypted))
            
            # Convert back to float (reinterpret encrypted bytes as floats)
            encrypted_array = np.frombuffer(encrypted, dtype=np.float32)
            encrypted_array = encrypted_array.reshape(buffer.data.shape)
            
            # Update buffer in-place
            buffer.data[:] = encrypted_array
            
            self.frames_processed += buffer.get_frame_count()
            return True
            
        except Exception as e:
            print(f"[FernetEncryptor] Encryption error: {e}")
            return False
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Fernet Encryptor",
            version="1.0.0",
            author="NDA Team",
            description="Fernet symmetric encryption (Python, EXAMPLE ONLY)",
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
        Set encryptor parameter.
        
        Supported parameters:
            - "key": Base64-encoded Fernet key (for sharing with decryptor)
        """
        if key == "key":
            try:
                self.key = value.encode() if isinstance(value, str) else value
                self.cipher = Fernet(self.key)
                print(f"[FernetEncryptor] Key updated: {self.key[:16]}...")
                return True
            except Exception as e:
                print(f"[FernetEncryptor] Invalid key: {e}")
                return False
        return False
    
    def get_parameter(self, key: str) -> str:
        """Get encryptor parameter"""
        if key == "key":
            return self.key.decode() if self.key else ""
        return ""


def create_plugin():
    """Factory function called by plugin loader"""
    return FernetEncryptorPlugin()


# Test standalone
if __name__ == "__main__":
    if not CRYPTO_AVAILABLE:
        print("Skipping test - cryptography library not available")
        sys.exit(1)
    
    plugin = create_plugin()
    assert plugin.initialize()
    assert plugin.start()
    
    # Test encryption
    buffer = AudioBuffer(2, 512)
    buffer.data = np.random.randn(2, 512).astype(np.float32)
    original_data = buffer.data.copy()
    
    assert plugin.process_audio(buffer)
    
    # Encrypted data should be different from original
    assert not np.array_equal(buffer.data, original_data), "Encryption should modify data!"
    
    plugin.stop()
    plugin.shutdown()
    
    print("✓ Fernet encryptor plugin test passed!")

