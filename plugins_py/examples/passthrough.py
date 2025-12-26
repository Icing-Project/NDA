"""
Passthrough Processor Plugin
No-op processor for testing pipeline integrity.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState, AudioBuffer


class PassthroughPlugin(AudioProcessorPlugin):
    """
    No-op processor that passes audio through unchanged.
    Used for testing pipeline integrity and as a template for processor development.
    """
    
    def __init__(self):
        super().__init__()
        self.state = PluginState.UNLOADED
        self.sample_rate = 48000
        self.channels = 2
    
    def initialize(self) -> bool:
        """Initialize processor"""
        self.state = PluginState.INITIALIZED
        print(f"[PassthroughPlugin] Initialized at {self.sample_rate}Hz, {self.channels} channels")
        return True
    
    def start(self) -> bool:
        """Start processing"""
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        print("[PassthroughPlugin] Started")
        return True
    
    def stop(self):
        """Stop processing"""
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED
            print("[PassthroughPlugin] Stopped")
    
    def shutdown(self):
        """Shutdown processor"""
        self.state = PluginState.UNLOADED
        print("[PassthroughPlugin] Shutdown")
    
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Pass audio through unchanged (no processing).
        
        Args:
            buffer: AudioBuffer to process (not modified)
        
        Returns:
            True (always succeeds)
        """
        if self.state != PluginState.RUNNING:
            return False
        
        # Intentionally empty - no processing
        # This is useful for testing that pipeline works without any processing
        return True
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Passthrough",
            version="1.0.0",
            author="NDA Team",
            description="No-op processor for pipeline testing",
            plugin_type=PluginType.PROCESSOR,
            api_version=1
        )
    
    def get_type(self) -> PluginType:
        """Get plugin type"""
        return PluginType.PROCESSOR
    
    def get_state(self) -> PluginState:
        """Get current state"""
        return self.state
    
    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate
    
    def get_channel_count(self) -> int:
        """Get channel count"""
        return self.channels
    
    def set_sample_rate(self, rate: int) -> None:
        """Set sample rate"""
        self.sample_rate = rate
    
    def set_channel_count(self, channels: int) -> None:
        """Set channel count"""
        self.channels = channels
    
    def set_parameter(self, key: str, value: str) -> bool:
        """Set parameter (none supported)"""
        return False
    
    def get_parameter(self, key: str) -> str:
        """Get parameter (none supported)"""
        return ""


def create_plugin():
    """Factory function called by plugin loader"""
    return PassthroughPlugin()


# Test standalone
if __name__ == "__main__":
    import numpy as np
    
    plugin = create_plugin()
    assert plugin.initialize()
    assert plugin.start()
    
    # Test with dummy buffer
    buffer = AudioBuffer(2, 512)
    buffer.data = np.random.randn(2, 512).astype(np.float32)
    original_data = buffer.data.copy()
    
    assert plugin.process_audio(buffer)
    
    # Data should be unchanged
    assert np.array_equal(buffer.data, original_data), "Passthrough should not modify data!"
    
    plugin.stop()
    plugin.shutdown()
    
    print("✓ Passthrough plugin test passed!")

