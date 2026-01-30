/**
 * @file TextEncoding.h
 * @brief Text encoding utilities for NDA-Nade text messaging integration.
 *
 * This header provides functions to encode text messages as "fake audio" samples
 * that can flow through NDA's audio pipeline. The encoding uses a magic number (π)
 * to identify text data and stores UTF-8 bytes as float32 values.
 *
 * Format:
 *   Sample[0]:     3.14159 (π) - Magic number to identify text data
 *   Sample[1]:     Length of text in bytes (N)
 *   Sample[2:N+2]: UTF-8 bytes as float32 values (0-255 range)
 *
 * @version 3.0
 * @date 2026-01-30
 */

#pragma once

#include <string>
#include <vector>
#include <optional>
#include <cmath>
#include <cstdint>

namespace nade {

/// Magic number (π) used to identify text-encoded audio data
constexpr float MAGIC_NUMBER = 3.14159f;

/// Tolerance for comparing magic number (accounts for float precision)
constexpr float MAGIC_TOLERANCE = 0.001f;

/// Maximum message length in characters (UTF-8)
constexpr size_t MAX_MESSAGE_LENGTH = 256;

/**
 * @brief Encode text as fake audio samples for NDA pipeline.
 *
 * @param text UTF-8 text to encode (max 256 characters)
 * @return std::vector<float> Audio buffer containing encoded text
 *
 * @example
 *   auto audio = textToAudioBuffer("Hi");
 *   // audio = [3.14159, 2.0, 72.0, 105.0]
 */
inline std::vector<float> textToAudioBuffer(const std::string& text) {
    const size_t length = text.size();

    // Allocate: magic + length + text bytes
    std::vector<float> audio(length + 2);

    // Magic identifier
    audio[0] = MAGIC_NUMBER;

    // Length field
    audio[1] = static_cast<float>(length);

    // Copy UTF-8 bytes as float values
    for (size_t i = 0; i < length; ++i) {
        audio[i + 2] = static_cast<float>(static_cast<uint8_t>(text[i]));
    }

    return audio;
}

/**
 * @brief Decode text from fake audio samples.
 *
 * @param audio Pointer to audio buffer
 * @param size Number of samples in buffer
 * @return std::optional<std::string> Decoded text, or nullopt if not valid text data
 *
 * @example
 *   float audio[] = {3.14159f, 2.0f, 72.0f, 105.0f};
 *   auto text = audioBufferToText(audio, 4);
 *   // text = "Hi"
 */
inline std::optional<std::string> audioBufferToText(const float* audio, size_t size) {
    // Minimum size: magic + length
    if (size < 2) {
        return std::nullopt;
    }

    // Check magic number (with tolerance for float precision)
    if (std::abs(audio[0] - MAGIC_NUMBER) > MAGIC_TOLERANCE) {
        return std::nullopt;  // Not text data
    }

    // Extract length
    const auto length = static_cast<size_t>(audio[1]);

    // Validate length
    if (length == 0) {
        return std::string{};  // Empty text is valid
    }

    if (length + 2 > size) {
        return std::nullopt;  // Malformed: not enough data
    }

    if (length > MAX_MESSAGE_LENGTH) {
        return std::nullopt;  // Too long
    }

    // Extract UTF-8 bytes
    std::string text;
    text.reserve(length);
    for (size_t i = 0; i < length; ++i) {
        text += static_cast<char>(static_cast<uint8_t>(audio[i + 2]));
    }

    return text;
}

/**
 * @brief Decode text from a vector of audio samples.
 *
 * Convenience overload that takes a vector reference.
 *
 * @param audio Vector of audio samples
 * @return std::optional<std::string> Decoded text, or nullopt if not valid text data
 */
inline std::optional<std::string> audioBufferToText(const std::vector<float>& audio) {
    return audioBufferToText(audio.data(), audio.size());
}

/**
 * @brief Check if audio buffer contains encoded text.
 *
 * @param audio Pointer to audio buffer
 * @param size Number of samples in buffer
 * @return true if buffer appears to contain text data
 */
inline bool isTextEncodedAudio(const float* audio, size_t size) {
    if (size < 2) {
        return false;
    }
    return std::abs(audio[0] - MAGIC_NUMBER) <= MAGIC_TOLERANCE;
}

/**
 * @brief Check if audio buffer contains encoded text.
 *
 * Convenience overload that takes a vector reference.
 *
 * @param audio Vector of audio samples
 * @return true if buffer appears to contain text data
 */
inline bool isTextEncodedAudio(const std::vector<float>& audio) {
    return isTextEncodedAudio(audio.data(), audio.size());
}

/**
 * @brief Get the expected buffer size for a given text length.
 *
 * @param textLength Length of text in bytes
 * @return size_t Number of float samples needed
 */
inline size_t getEncodedBufferSize(size_t textLength) {
    return textLength + 2;  // magic + length + text bytes
}

}  // namespace nade
