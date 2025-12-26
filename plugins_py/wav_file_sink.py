"""
WAV File Sink Plugin - Python Implementation
Records audio to WAV file (32-bit float PCM)
"""

import wave
import struct
import numpy as np
from datetime import datetime
from base_plugin import (
    AudioSinkPlugin, AudioBuffer, PluginInfo,
    PluginType, PluginState
)


class WavFileSinkPlugin(AudioSinkPlugin):
    """Records audio to WAV file"""

    def __init__(self):
        super().__init__()
        self.sample_rate = 48000
        self.channel_count = 2
        self.buffer_size = 512
        self.total_frames = 0
        self.wav_file = None
        self.current_filename = ""
        self.custom_filename = ""

    def initialize(self) -> bool:
        """Initialize the plugin"""
        self.state = PluginState.INITIALIZED
        return True

    def shutdown(self):
        """Shutdown the plugin"""
        self.stop()
        self.state = PluginState.UNLOADED

    def start(self) -> bool:
        """Start recording"""
        if self.state != PluginState.INITIALIZED:
            return False

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.custom_filename:
            self.current_filename = self.custom_filename
        else:
            self.current_filename = f"recording_{timestamp}.wav"

        try:
            # Open WAV file (we'll use scipy.io.wavfile for 32-bit float support)
            # For now, use a manual implementation
            self.wav_file = open(self.current_filename, 'wb', buffering=8192)
            self._write_wav_header(0)

            self.total_frames = 0
            self.state = PluginState.RUNNING

            print(f"[WavFileSink] Recording to: {self.current_filename}", flush=True)
            print(f"[WavFileSink] Format: {self.sample_rate} Hz, {self.channel_count} channels, 32-bit float", flush=True)

            return True
        except Exception as e:
            print(f"[WavFileSink] Failed to open file: {self.current_filename} - {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        """Stop recording"""
        if self.state != PluginState.RUNNING:
            return

        try:
            if self.wav_file:
                # Update WAV header with correct size
                self.wav_file.seek(0)
                self._write_wav_header(self.total_frames)
                self.wav_file.close()
                self.wav_file = None

                duration = self.total_frames / self.sample_rate
                print(f"[WavFileSink] Recording stopped", flush=True)
                print(f"[WavFileSink] Saved {self.total_frames} frames ({duration:.2f} seconds) to {self.current_filename}", flush=True)
        except Exception as e:
            print(f"[WavFileSink] Error closing file: {e}", flush=True)
            if self.wav_file:
                try:
                    self.wav_file.close()
                except:
                    pass
                self.wav_file = None

        self.state = PluginState.INITIALIZED

    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        return PluginInfo(
            name="WAV File Recorder",
            version="1.0.0",
            author="Icing Project",
            description="Records audio to WAV file (32-bit float PCM)",
            plugin_type=PluginType.AUDIO_SINK,
            api_version=1
        )

    def set_parameter(self, key: str, value: str):
        """Set plugin parameter"""
        if key == "filename":
            self.custom_filename = value

    def get_parameter(self, key: str) -> str:
        """Get plugin parameter"""
        if key == "filename":
            return self.current_filename
        elif key == "sampleRate":
            return str(self.sample_rate)
        elif key == "channels":
            return str(self.channel_count)
        return ""

    def write_audio(self, buffer: AudioBuffer) -> bool:
        """Write audio data to file"""
        if not self.wav_file or self.state != PluginState.RUNNING:
            return False

        try:
            frame_count = buffer.get_frame_count()
            channel_count = buffer.get_channel_count()

            # Interleave samples - use numpy's efficient reshaping
            # buffer.data has shape (channels, frames)
            # We need to transpose and flatten to get interleaved format
            interleaved = buffer.data.T.flatten().astype(np.float32)

            # Write to file
            self.wav_file.write(interleaved.tobytes())
            self.wav_file.flush()  # Ensure data is written

            self.total_frames += frame_count

            # Print progress every second
            if self.total_frames % self.sample_rate < frame_count:
                seconds = self.total_frames / self.sample_rate
                print(f"[WavFileSink] Recording: {seconds:.0f}s", flush=True)

            return True
        except Exception as e:
            print(f"[WavFileSink] Error writing audio: {e}", flush=True)
            import traceback
            traceback.print_exc()
            return False

    def _write_wav_header(self, total_frames: int):
        """Write WAV file header"""
        data_size = total_frames * self.channel_count * 4  # 4 bytes per float32
        file_size = 36 + data_size

        # RIFF header
        self.wav_file.write(b'RIFF')
        self.wav_file.write(struct.pack('<I', file_size))
        self.wav_file.write(b'WAVE')

        # fmt chunk
        self.wav_file.write(b'fmt ')
        self.wav_file.write(struct.pack('<I', 16))  # fmt chunk size
        self.wav_file.write(struct.pack('<H', 3))   # audio format (3 = IEEE float)
        self.wav_file.write(struct.pack('<H', self.channel_count))
        self.wav_file.write(struct.pack('<I', self.sample_rate))
        self.wav_file.write(struct.pack('<I', self.sample_rate * self.channel_count * 4))  # byte rate
        self.wav_file.write(struct.pack('<H', self.channel_count * 4))  # block align
        self.wav_file.write(struct.pack('<H', 32))  # bits per sample

        # data chunk
        self.wav_file.write(b'data')
        self.wav_file.write(struct.pack('<I', data_size))

    def get_sample_rate(self) -> int:
        """Get sample rate"""
        return self.sample_rate

    def get_channel_count(self) -> int:
        """Get number of channels"""
        return self.channel_count

    def set_sample_rate(self, sample_rate: int):
        """Set sample rate"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.sample_rate = sample_rate

    def set_channel_count(self, channels: int):
        """Set number of channels"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.channel_count = channels

    def get_buffer_size(self) -> int:
        """Get buffer size"""
        return self.buffer_size

    def set_buffer_size(self, samples: int):
        """Set buffer size"""
        if self.state in (PluginState.UNLOADED, PluginState.INITIALIZED):
            self.buffer_size = samples

    def get_available_space(self) -> int:
        """Get available space (plenty of space)"""
        return 1000000


# Plugin factory function
def create_plugin():
    """Factory function to create plugin instance"""
    return WavFileSinkPlugin()
