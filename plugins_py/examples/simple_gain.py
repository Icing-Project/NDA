"""
Simple Gain Processor Plugin
Adjusts audio volume/gain with parameter control.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from base_plugin import AudioProcessorPlugin, PluginInfo, PluginType, PluginState, AudioBuffer


class SimpleGainPlugin(AudioProcessorPlugin):
    """
    Basic volume/gain adjustment processor.
    
    Demonstrates parameter handling and in-place audio processing.
    Supports gain values from 0.0 (mute) to 2.0 (200% volume).
    """
    
    def __init__(self):
        super().__init__()
        self.state = PluginState.UNLOADED
        self.gain = 1.0  # Unity gain (no change) by default
        self.sample_rate = 48000
        self.channels = 2
        self.frames_processed = 0
    
    def initialize(self) -> bool:
        """Initialize processor"""
        self.state = PluginState.INITIALIZED
        self.frames_processed = 0
        print(f"[SimpleGainPlugin] Initialized at {self.sample_rate}Hz, "
              f"{self.channels} channels, gain={self.gain}")
        return True
    
    def start(self) -> bool:
        """Start processing"""
        if self.state != PluginState.INITIALIZED:
            return False
        self.state = PluginState.RUNNING
        print(f"[SimpleGainPlugin] Started with gain={self.gain}")
        return True
    
    def stop(self):
        """Stop processing"""
        if self.state == PluginState.RUNNING:
            self.state = PluginState.INITIALIZED
            seconds = self.frames_processed / self.sample_rate
            print(f"[SimpleGainPlugin] Stopped after processing {self.frames_processed} frames ({seconds:.2f}s)")
    
    def shutdown(self):
        """Shutdown processor"""
        self.state = PluginState.UNLOADED
        print("[SimpleGainPlugin] Shutdown")
    
    def process_audio(self, buffer: AudioBuffer) -> bool:
        """
        Apply gain to audio buffer in-place.
        
        Args:
            buffer: AudioBuffer to process (modified in-place)
        
        Returns:
            True on success, False on error
        """
        if self.state != PluginState.RUNNING:
            return False
        
        try:
            # Apply gain to all samples
            buffer.data *= self.gain
            
            # Clip to prevent overflow (hard limiting)
            np.clip(buffer.data, -1.0, 1.0, out=buffer.data)
            
            self.frames_processed += buffer.get_frame_count()
            return True
            
        except Exception as e:
            print(f"[SimpleGainPlugin] Error processing audio: {e}")
            return False
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="Simple Gain",
            version="1.0.0",
            author="NDA Team",
            description="Basic volume/gain adjustment processor",
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
        """
        Set plugin parameter.
        
        Supported parameters:
            - "gain": float value (0.0 to 2.0)
        """
        if key == "gain":
            try:
                new_gain = float(value)
                if 0.0 <= new_gain <= 2.0:
                    self.gain = new_gain
                    print(f"[SimpleGainPlugin] Gain set to {self.gain}")
                    return True
                else:
                    print(f"[SimpleGainPlugin] Invalid gain value: {new_gain} (must be 0.0-2.0)")
                    return False
            except ValueError:
                print(f"[SimpleGainPlugin] Invalid gain format: {value}")
                return False
        
        return False
    
    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "gain":
            return str(self.gain)
        return ""


def create_plugin():
    """Factory function called by plugin loader"""
    return SimpleGainPlugin()


# Test standalone
if __name__ == "__main__":
    plugin = create_plugin()
    assert plugin.initialize()
    assert plugin.start()
    
    # Test with dummy buffer
    buffer = AudioBuffer(2, 512)
    buffer.data.fill(0.5)  # Half amplitude
    
    # Test unity gain (no change)
    assert plugin.set_parameter("gain", "1.0")
    assert plugin.process_audio(buffer)
    assert np.allclose(buffer.data, 0.5), "Unity gain should not change amplitude!"
    
    # Test 50% gain
    buffer.data.fill(0.5)
    assert plugin.set_parameter("gain", "0.5")
    assert plugin.process_audio(buffer)
    assert np.allclose(buffer.data, 0.25), "50% gain failed!"
    
    # Test 200% gain
    buffer.data.fill(0.3)
    assert plugin.set_parameter("gain", "2.0")
    assert plugin.process_audio(buffer)
    assert np.allclose(buffer.data, 0.6), "200% gain failed!"
    
    # Test clipping
    buffer.data.fill(0.8)
    assert plugin.set_parameter("gain", "2.0")
    assert plugin.process_audio(buffer)
    assert np.all(buffer.data <= 1.0), "Clipping failed - exceeded 1.0!"
    
    plugin.stop()
    plugin.shutdown()
    
    print("✓ SimpleGain plugin test passed!")

