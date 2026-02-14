cmake_minimum_required(VERSION 3.16)
project(NDA VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Option to enable Python plugin support
# Default: OFF for release builds (stability), set to ON for development
option(NDA_ENABLE_PYTHON "Enable Python plugin support" OFF)

# Optional overrides for the embedded Python used by `find_package(Python3 ...)`.
# Note: a virtualenv alone is often insufficient on Windows unless its base Python
# installation provides the Development files (headers/libs). NumPy must be
# importable by the selected interpreter at configure time.
set(NDA_PYTHON_EXECUTABLE "" CACHE FILEPATH "Python executable for embedding (must have NumPy installed)")
set(NDA_PYTHON_ROOT_DIR "" CACHE PATH "Python root directory for embedding")

# If no explicit Python override is provided, prefer the repo-local venv (if present).
set(_nda_python_autodetected FALSE)
if(WIN32)
    set(_nda_default_python_executable "")
    set(_nda_default_python_root_dir "")

    # Helper macro to parse pyvenv.cfg and extract the 'home' directory
    macro(_nda_parse_pyvenv_cfg cfg_path out_var)
        if(EXISTS "${cfg_path}")
            file(STRINGS "${cfg_path}" _nda_venv_home_lines REGEX "^home\\s*=")
            if(_nda_venv_home_lines)
                list(GET _nda_venv_home_lines 0 _nda_venv_home_line)
                string(REGEX REPLACE "^home\\s*=\\s*" "" ${out_var} "${_nda_venv_home_line}")
            endif()
        endif()
    endmacro()

    if(EXISTS "${CMAKE_SOURCE_DIR}/venv/Scripts/python.exe")
        set(_nda_default_python_executable "${CMAKE_SOURCE_DIR}/venv/Scripts/python.exe")
        _nda_parse_pyvenv_cfg("${CMAKE_SOURCE_DIR}/venv/pyvenv.cfg" _nda_default_python_root_dir)
    elseif(EXISTS "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
        set(_nda_default_python_executable "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
        _nda_parse_pyvenv_cfg("${CMAKE_SOURCE_DIR}/.venv/pyvenv.cfg" _nda_default_python_root_dir)
    endif()

    if(NDA_PYTHON_EXECUTABLE STREQUAL "" AND _nda_default_python_executable)
        set(NDA_PYTHON_EXECUTABLE "${_nda_default_python_executable}" CACHE FILEPATH "Python executable for embedding (must have NumPy installed)" FORCE)
        set(_nda_python_autodetected TRUE)
    endif()
    if(NDA_PYTHON_ROOT_DIR STREQUAL "" AND _nda_default_python_root_dir)
        set(NDA_PYTHON_ROOT_DIR "${_nda_default_python_root_dir}" CACHE PATH "Python root directory for embedding" FORCE)
        set(_nda_python_autodetected TRUE)
    endif()
endif()

if(NDA_PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE "${NDA_PYTHON_EXECUTABLE}")
endif()
if(NDA_PYTHON_ROOT_DIR)
    set(Python3_ROOT_DIR "${NDA_PYTHON_ROOT_DIR}")
endif()

# CLion (and other IDEs) may inject Python3_INCLUDE_DIR / Python3_LIBRARY_* for a different
# Python than the one selected above. When the NDA overrides are provided, prefer them and
# clear conflicting hints to avoid version-mismatch failures.
if(NDA_PYTHON_EXECUTABLE OR NDA_PYTHON_ROOT_DIR)
    unset(Python3_INCLUDE_DIR CACHE)
    unset(Python3_LIBRARY_RELEASE CACHE)
    unset(Python3_LIBRARY_DEBUG CACHE)
    unset(Python3_LIBRARY CACHE)

    # Prefer the explicitly-provided interpreter/root, and avoid picking up an unrelated registry install.
    set(Python3_FIND_STRATEGY LOCATION)
    set(Python3_FIND_VIRTUALENV FIRST)
    if(WIN32)
        set(Python3_FIND_REGISTRY NEVER)
    endif()
endif()

# Qt6 packages
# NDA does not use Vulkan directly; Qt may try to enable optional Vulkan support and emit
# warnings if Vulkan headers are not present. Disable it by default to keep configuration clean.
if(WIN32)
    set(CMAKE_DISABLE_FIND_PACKAGE_WrapVulkanHeaders TRUE)
endif()
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Network)

# OpenSSL for encryption (now plugin-only, not core dependency in v2.0)
find_package(OpenSSL REQUIRED)

# ===================================================================
# libnade_crypto - Shared Crypto Library
# ===================================================================
add_library(nade_crypto SHARED
    include/crypto/CryptoManager.h
    src/crypto/CryptoManager.cpp
)

target_include_directories(nade_crypto
    PUBLIC ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(nade_crypto
    PUBLIC
        OpenSSL::SSL
        OpenSSL::Crypto
        Qt6::Core
)

target_compile_definitions(nade_crypto
    PRIVATE NADE_CRYPTO_EXPORTS
)

set_target_properties(nade_crypto PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# Install shared library
install(TARGETS nade_crypto
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Optional: libsamplerate for high-quality resampling (primarily via pkg-config on Unix).
if(NOT WIN32)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(SAMPLERATE samplerate)
        if(SAMPLERATE_FOUND)
            message(STATUS "libsamplerate found - High quality resampling available")
            add_compile_definitions(HAVE_LIBSAMPLERATE)
            include_directories(${SAMPLERATE_INCLUDE_DIRS})
        else()
            message(STATUS "libsamplerate not found - Using Simple/Medium quality only")
        endif()

        # Linux audio support via PulseAudio
        pkg_check_modules(PULSEAUDIO libpulse)
        if(PULSEAUDIO_FOUND)
            message(STATUS "PulseAudio found - Linux audio plugins available")
            message(STATUS "  PulseAudio version: ${PULSEAUDIO_VERSION}")
            set(LINUX_AUDIO_AVAILABLE TRUE)
        else()
            message(STATUS "PulseAudio not found - Linux audio plugins disabled")
            message(STATUS "  Install with: sudo dnf install pulseaudio-libs-devel (Fedora)")
            message(STATUS "             or: sudo apt install libpulse-dev (Ubuntu/Debian)")
            set(LINUX_AUDIO_AVAILABLE FALSE)
        endif()

        # Linux HID support via hidapi (for AIOC PTT control)
        pkg_check_modules(HIDAPI hidapi-libusb)
        if(NOT HIDAPI_FOUND)
            # Try alternative package name (some distros use hidapi instead of hidapi-libusb)
            pkg_check_modules(HIDAPI hidapi)
        endif()
        if(HIDAPI_FOUND)
            message(STATUS "hidapi found - Linux AIOC PTT control available")
            message(STATUS "  hidapi version: ${HIDAPI_VERSION}")
            set(LINUX_HIDAPI_AVAILABLE TRUE)
        else()
            message(STATUS "hidapi not found - Linux AIOC PTT control disabled")
            message(STATUS "  Install with: sudo dnf install hidapi-devel (Fedora)")
            message(STATUS "             or: sudo apt install libhidapi-dev (Ubuntu/Debian)")
            set(LINUX_HIDAPI_AVAILABLE FALSE)
        endif()
    else()
        message(STATUS "pkg-config not found - libsamplerate detection skipped")
        set(LINUX_AUDIO_AVAILABLE FALSE)
        set(LINUX_HIDAPI_AVAILABLE FALSE)
    endif()
else()
    message(STATUS "pkg-config skipped on Windows (libsamplerate optional)")
    set(LINUX_AUDIO_AVAILABLE FALSE)
    set(LINUX_HIDAPI_AVAILABLE FALSE)
endif()

# Find Python and NumPy if Python support is enabled
if(NDA_ENABLE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development NumPy)
    if(Python3_FOUND)
        message(STATUS "Python support enabled")
        message(STATUS "  Python executable: ${Python3_EXECUTABLE}")
        message(STATUS "  Python version: ${Python3_VERSION}")
        message(STATUS "  Python include dirs: ${Python3_INCLUDE_DIRS}")
        message(STATUS "  NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")

        # Many Windows Python distributions (including uv-managed CPython) do not ship a separate
        # debug import library (python3x_d.lib). For MSVC Debug builds, fall back to the Release
        # import library so developers can still build/run NDA in Debug.
        if(WIN32 AND TARGET Python3::Python)
            get_target_property(_py_implib_release Python3::Python IMPORTED_IMPLIB_RELEASE)
            get_target_property(_py_implib_debug Python3::Python IMPORTED_IMPLIB_DEBUG)
            if(_py_implib_release AND EXISTS "${_py_implib_release}")
                if((NOT _py_implib_debug) OR (NOT EXISTS "${_py_implib_debug}"))
                    message(STATUS "  Python debug import library not found; using release import library for Debug config")
                    set_property(TARGET Python3::Python PROPERTY IMPORTED_IMPLIB_DEBUG "${_py_implib_release}")
                endif()
            endif()
        endif()

        add_compile_definitions(NDA_ENABLE_PYTHON)

        # Find pybind11 for embedding Python in plugins
        find_package(pybind11 CONFIG QUIET)
        if(pybind11_FOUND)
            message(STATUS "  pybind11 found via CMake config: ${pybind11_VERSION}")
            set(NDA_PYBIND11_FOUND TRUE)
        else()
            # Try to get pybind11 include path from Python package
            execute_process(
                COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
                OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE PYBIND11_RESULT
            )
            if(PYBIND11_RESULT EQUAL 0 AND EXISTS "${PYBIND11_INCLUDE_DIR}")
                message(STATUS "  pybind11 found via Python package: ${PYBIND11_INCLUDE_DIR}")
                set(NDA_PYBIND11_FOUND TRUE)
                set(NDA_PYBIND11_INCLUDE_DIR "${PYBIND11_INCLUDE_DIR}")
            else()
                message(STATUS "  pybind11 not found - Nade Python plugins will be disabled")
                message(STATUS "  Install with: pip install pybind11")
                set(NDA_PYBIND11_FOUND FALSE)
            endif()
        endif()
    else()
        if(Python3_Interpreter_FOUND AND NOT Python3_Development_FOUND)
            message(WARNING "Python interpreter found but Development files not found; point CMake to a full Python install (or its base), not just a venv")
        endif()
        if(Python3_Interpreter_FOUND AND NOT Python3_NumPy_FOUND)
            message(WARNING "Python interpreter found but NumPy not found; install it into the selected interpreter before configuring (e.g. `pip install numpy`)")
        endif()
        message(WARNING "Python3 not found, Python plugin support disabled")
        set(NDA_ENABLE_PYTHON OFF)
    endif()
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files (v2.0: Unified view)
set(SOURCES
    src/main.cpp
    src/ui/MainWindow.cpp
    src/ui/UnifiedPipelineView.cpp
    src/ui/PluginSidebar.cpp
    src/ui/AboutDialog.cpp
    src/ui/ImportKeysDialog.cpp
    src/ui/ExportKeysDialog.cpp
    # v2.2: Removed dead UI views (AudioDevicesView, EncryptionView, PluginsView, SettingsView)
    src/audio/AudioEngine.cpp
    src/audio/AudioDevice.cpp
    src/audio/Resampler.cpp
    src/audio/WasapiDeviceEnum.cpp
    src/core/ProcessingPipeline.cpp
    src/plugins/PluginManager.cpp
    src/plugins/PluginPaths.cpp
)

# Add Python bridge if enabled
if(NDA_ENABLE_PYTHON)
    list(APPEND SOURCES src/plugins/PythonPluginBridge.cpp)
endif()

# Add Linux audio device enumeration if PulseAudio is available
if(LINUX_AUDIO_AVAILABLE)
    list(APPEND SOURCES src/audio/PulseDeviceEnum.cpp)
    list(APPEND HEADERS include/audio/PulseDeviceEnum.h)
else()
    # Add stub implementations when PulseAudio is not available
    list(APPEND SOURCES src/audio/PulseDeviceEnumStub.cpp)
    list(APPEND HEADERS include/audio/PulseDeviceEnum.h)
endif()

# Header files (v2.0: Unified view)
set(HEADERS
    include/ui/MainWindow.h
    include/ui/UnifiedPipelineView.h
    include/ui/PluginSidebar.h
    include/ui/AboutDialog.h
    include/ui/ImportKeysDialog.h
    include/ui/ExportKeysDialog.h
    # v2.2: Removed dead UI views (see SOURCES comment)
    include/audio/AudioEngine.h
    include/audio/AudioDevice.h
    include/audio/AudioBuffer.h
    include/audio/Resampler.h
    include/audio/RingBuffer.h
    include/audio/WasapiDeviceEnum.h
    include/core/ProcessingPipeline.h
    include/plugins/PluginTypes.h
    include/plugins/BasePlugin.h
    include/plugins/AudioSourcePlugin.h
    include/plugins/AudioProcessorPlugin.h
    include/plugins/AudioSinkPlugin.h
    include/plugins/PluginManager.h
    include/plugins/PluginPaths.h
    include/crypto/CryptoManager.h
)

# Add Python bridge header if enabled
if(NDA_ENABLE_PYTHON)
    list(APPEND HEADERS include/plugins/PythonPluginBridge.h)
endif()

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# On ELF platforms, export executable symbols so runtime-loaded plugins can
# resolve references to core functions compiled into the main binary.
if(UNIX AND NOT APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES ENABLE_EXPORTS ON)
endif()

# Link libraries
target_link_libraries(${PROJECT_NAME}
    nade_crypto
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Network
)

# Link libsamplerate if available
if(SAMPLERATE_FOUND)
    target_link_libraries(${PROJECT_NAME} ${SAMPLERATE_LIBRARIES})
endif()

# Link PulseAudio if available (for device enumeration)
if(LINUX_AUDIO_AVAILABLE)
    target_include_directories(${PROJECT_NAME} PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} ${PULSEAUDIO_LIBRARIES})
endif()

# Link Python libraries if enabled
if(NDA_ENABLE_PYTHON)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} Python3::Python)

    # On Windows, copy the Python DLL to the build output directory.
    # UV-managed venvs (and some other venvs) don't include the DLL, so we need
    # to copy it from the base Python installation.
    if(WIN32)
        # Determine the Python DLL name (e.g., python312.dll)
        set(_nda_python_dll_name "python${Python3_VERSION_MAJOR}${Python3_VERSION_MINOR}.dll")

        # Find the DLL in these locations (in order of preference):
        # 1. Python3_RUNTIME_LIBRARY_DIRS (if set by find_package)
        # 2. Python3_ROOT_DIR (base Python installation)
        # 3. Directory containing Python3_EXECUTABLE
        set(_nda_python_dll_search_paths
            ${Python3_RUNTIME_LIBRARY_DIRS}
            "${Python3_ROOT_DIR}"
            "${NDA_PYTHON_ROOT_DIR}"
        )
        # Add the directory containing the Python executable
        get_filename_component(_nda_python_exe_dir "${Python3_EXECUTABLE}" DIRECTORY)
        list(APPEND _nda_python_dll_search_paths "${_nda_python_exe_dir}")

        # Search for the DLL
        unset(_nda_python_dll_path CACHE)
        find_file(_nda_python_dll_path
            NAMES "${_nda_python_dll_name}"
            PATHS ${_nda_python_dll_search_paths}
            NO_DEFAULT_PATH
        )

        if(_nda_python_dll_path)
            message(STATUS "  Python DLL found: ${_nda_python_dll_path}")
            message(STATUS "  Will copy to build output directory")

            # Copy DLL to each configuration's output directory
            add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${_nda_python_dll_path}"
                    "$<TARGET_FILE_DIR:${PROJECT_NAME}>/${_nda_python_dll_name}"
                COMMENT "Copying ${_nda_python_dll_name} to output directory"
            )
        else()
            message(WARNING "  Python DLL (${_nda_python_dll_name}) not found in: ${_nda_python_dll_search_paths}")
            message(WARNING "  You may need to manually copy it to the build output directory or add its location to PATH")
        endif()
    endif()
endif()

# Plugin builds (shared libraries placed under build/plugins)
set(PLUGIN_OUTPUT_DIR ${CMAKE_BINARY_DIR}/plugins)

function(add_nda_plugin target)
    add_library(${target} SHARED ${ARGN})
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/plugins_src
    )
    if(WIN32)
        target_compile_definitions(${target} PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    endif()
    # Force flat plugin directory (no Release/Debug subdirs)
    # This makes multi-config generators (Visual Studio) behave like single-config
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
        # Override per-config directories to same location
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PLUGIN_OUTPUT_DIR}
        RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${PLUGIN_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
    )
endfunction()

set(NDA_PLUGIN_TARGETS)

if(WIN32)
    add_nda_plugin(AIOCSourcePlugin
        plugins_src/AIOCPluginCommon.cpp
        plugins_src/AIOCSourcePlugin.cpp
    )
    target_compile_definitions(AIOCSourcePlugin PRIVATE INITGUID)
    target_link_libraries(AIOCSourcePlugin ole32 avrt setupapi hid propsys)
    list(APPEND NDA_PLUGIN_TARGETS AIOCSourcePlugin)

    add_nda_plugin(AIOCSinkPlugin
        plugins_src/AIOCPluginCommon.cpp
        plugins_src/AIOCSinkPlugin.cpp
    )
    target_compile_definitions(AIOCSinkPlugin PRIVATE INITGUID)
    target_link_libraries(AIOCSinkPlugin ole32 avrt setupapi hid propsys)
    list(APPEND NDA_PLUGIN_TARGETS AIOCSinkPlugin)

    # Windows speaker/microphone plugins (WASAPI)
    add_nda_plugin(WindowsSpeakerSinkPlugin
        plugins_src/WindowsSpeakerSinkPlugin.cpp
    )
    target_compile_definitions(WindowsSpeakerSinkPlugin PRIVATE INITGUID)
    target_link_libraries(WindowsSpeakerSinkPlugin ole32 ksuser)
    list(APPEND NDA_PLUGIN_TARGETS WindowsSpeakerSinkPlugin)

    add_nda_plugin(WindowsMicrophoneSourcePlugin
        plugins_src/WindowsMicrophoneSourcePlugin.cpp
    )
    target_compile_definitions(WindowsMicrophoneSourcePlugin PRIVATE INITGUID)
    target_link_libraries(WindowsMicrophoneSourcePlugin ole32 ksuser)
    list(APPEND NDA_PLUGIN_TARGETS WindowsMicrophoneSourcePlugin)
endif()

add_nda_plugin(SineWaveSourcePlugin plugins_src/SineWaveSourcePlugin.cpp)
add_nda_plugin(NullSinkPlugin plugins_src/NullSinkPlugin.cpp)
add_nda_plugin(WavFileSinkPlugin plugins_src/WavFileSinkPlugin.cpp)
target_compile_definitions(SineWaveSourcePlugin PRIVATE _USE_MATH_DEFINES)
list(APPEND NDA_PLUGIN_TARGETS
    SineWaveSourcePlugin
    NullSinkPlugin
    WavFileSinkPlugin
)

# Cross-platform Pipeline Bridge plugins (zero-latency inter-pipeline transfer)
# The session is a shared library so both plugins share the same singleton instance
add_library(PipeBridgeSession SHARED
    plugins_src/bridge/PipeBridgeSession.cpp
)
target_include_directories(PipeBridgeSession PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_definitions(PipeBridgeSession PRIVATE PIPEBRIDGE_SESSION_EXPORTS)
set_target_properties(PipeBridgeSession PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
)

add_nda_plugin(PipeBridgeSinkPlugin
    plugins_src/bridge/PipeBridgeSink.cpp
)
target_link_libraries(PipeBridgeSinkPlugin PipeBridgeSession)

add_nda_plugin(PipeBridgeSourcePlugin
    plugins_src/bridge/PipeBridgeSource.cpp
)
target_link_libraries(PipeBridgeSourcePlugin PipeBridgeSession)

list(APPEND NDA_PLUGIN_TARGETS
    PipeBridgeSinkPlugin
    PipeBridgeSourcePlugin
)

# Linux audio plugins (PulseAudio/PipeWire)
if(LINUX_AUDIO_AVAILABLE)
    add_nda_plugin(PulseAudioSourcePlugin
        plugins_src/linux/PulseAudioSource.cpp
    )
    target_include_directories(PulseAudioSourcePlugin PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})
    target_link_libraries(PulseAudioSourcePlugin ${PULSEAUDIO_LIBRARIES})
    list(APPEND NDA_PLUGIN_TARGETS PulseAudioSourcePlugin)

    add_nda_plugin(PulseAudioSinkPlugin
        plugins_src/linux/PulseAudioSink.cpp
    )
    target_include_directories(PulseAudioSinkPlugin PRIVATE ${PULSEAUDIO_INCLUDE_DIRS})
    target_link_libraries(PulseAudioSinkPlugin ${PULSEAUDIO_LIBRARIES})
    list(APPEND NDA_PLUGIN_TARGETS PulseAudioSinkPlugin)

    message(STATUS "Linux audio plugins enabled: PulseAudioSourcePlugin, PulseAudioSinkPlugin")

    # Linux AIOC plugins (PulseAudio + hidapi for PTT)
    if(LINUX_HIDAPI_AVAILABLE)
        add_nda_plugin(LinuxAIOCSourcePlugin
            plugins_src/linux/LinuxAIOCSourcePlugin.cpp
            plugins_src/linux/LinuxAIOCSession.cpp
        )
        target_include_directories(LinuxAIOCSourcePlugin PRIVATE
            ${PULSEAUDIO_INCLUDE_DIRS}
            ${HIDAPI_INCLUDE_DIRS}
        )
        target_link_libraries(LinuxAIOCSourcePlugin
            ${PULSEAUDIO_LIBRARIES}
            ${HIDAPI_LIBRARIES}
        )
        list(APPEND NDA_PLUGIN_TARGETS LinuxAIOCSourcePlugin)

        add_nda_plugin(LinuxAIOCSinkPlugin
            plugins_src/linux/LinuxAIOCSinkPlugin.cpp
            plugins_src/linux/LinuxAIOCSession.cpp
        )
        target_include_directories(LinuxAIOCSinkPlugin PRIVATE
            ${PULSEAUDIO_INCLUDE_DIRS}
            ${HIDAPI_INCLUDE_DIRS}
        )
        target_link_libraries(LinuxAIOCSinkPlugin
            ${PULSEAUDIO_LIBRARIES}
            ${HIDAPI_LIBRARIES}
        )
        list(APPEND NDA_PLUGIN_TARGETS LinuxAIOCSinkPlugin)

        message(STATUS "Linux AIOC plugins enabled: LinuxAIOCSourcePlugin, LinuxAIOCSinkPlugin")
    else()
        message(STATUS "Linux AIOC plugins disabled (hidapi not found)")
    endif()
endif()

# Add example processor plugins (AES256 Encryptor/Decryptor)
# add_subdirectory(plugins_src/examples)

# ============================================================================
# Nade Text Messaging Plugins (encrypted text over FSK radio)
# Requires: Qt6, Python3 (with pybind11 for NDA_ENABLE_PYTHON)
# ============================================================================

# Nade common library (shared between plugins)
# IMPORTANT: This MUST be a SHARED library so all Nade plugins share the same
# singleton instance of NadeExternalIO. If this were STATIC, each plugin DLL
# would get its own copy of the singleton, causing multiple Python interpreter
# initializations which crashes with "The interpreter is already running".
# (See PipeBridgeSession for the same pattern.)
add_library(NadeCommon SHARED
    plugins_src/nade/NadeExternalIO.cpp
    plugins_src/nade/NadeExternalIO.h
    plugins_src/nade/TextEncoding.h
)
target_include_directories(NadeCommon PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/plugins_src
)
target_compile_definitions(NadeCommon PRIVATE NADE_COMMON_EXPORTS)
if(WIN32)
    target_compile_definitions(NadeCommon PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()
set_target_properties(NadeCommon PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PLUGIN_OUTPUT_DIR}
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${PLUGIN_OUTPUT_DIR}
)
target_link_libraries(NadeCommon
    nade_crypto
    Qt6::Widgets
)
if(NDA_ENABLE_PYTHON AND Python3_FOUND AND NDA_PYBIND11_FOUND)
    target_compile_definitions(NadeCommon PRIVATE NDA_ENABLE_PYTHON)
    target_include_directories(NadeCommon PRIVATE ${Python3_INCLUDE_DIRS})
    if(pybind11_FOUND)
        # pybind11 found via CMake config - use target
        target_link_libraries(NadeCommon pybind11::embed)
    else()
        # pybind11 found via Python - include manually
        target_include_directories(NadeCommon PRIVATE ${NDA_PYBIND11_INCLUDE_DIR})
        target_link_libraries(NadeCommon Python3::Python)
    endif()
endif()

# Nade Text Source Plugin (TX path - text input GUI)
add_nda_plugin(NadeTextSourcePlugin
    plugins_src/nade/NadeTextSourcePlugin.cpp
    plugins_src/nade/NadeTextSourcePlugin.h
)
target_include_directories(NadeTextSourcePlugin PRIVATE ${CMAKE_SOURCE_DIR}/plugins_src)
target_link_libraries(NadeTextSourcePlugin NadeCommon Qt6::Widgets)
list(APPEND NDA_PLUGIN_TARGETS NadeTextSourcePlugin)

# Nade Encryptor Plugin (TX path - processor)
add_nda_plugin(NadeEncryptorPlugin
    plugins_src/nade/NadeEncryptorPlugin.cpp
    plugins_src/nade/NadeEncryptorPlugin.h
)
target_include_directories(NadeEncryptorPlugin PRIVATE ${CMAKE_SOURCE_DIR}/plugins_src)
target_link_libraries(NadeEncryptorPlugin NadeCommon)
list(APPEND NDA_PLUGIN_TARGETS NadeEncryptorPlugin)

# Nade Decryptor Plugin (RX path - processor)
add_nda_plugin(NadeDecryptorPlugin
    plugins_src/nade/NadeDecryptorPlugin.cpp
    plugins_src/nade/NadeDecryptorPlugin.h
)
target_include_directories(NadeDecryptorPlugin PRIVATE ${CMAKE_SOURCE_DIR}/plugins_src)
target_link_libraries(NadeDecryptorPlugin NadeCommon)
list(APPEND NDA_PLUGIN_TARGETS NadeDecryptorPlugin)

# Nade Text Sink Plugin (RX path - text display GUI)
add_nda_plugin(NadeTextSinkPlugin
    plugins_src/nade/NadeTextSinkPlugin.cpp
    plugins_src/nade/NadeTextSinkPlugin.h
)
target_include_directories(NadeTextSinkPlugin PRIVATE ${CMAKE_SOURCE_DIR}/plugins_src)
target_link_libraries(NadeTextSinkPlugin NadeCommon Qt6::Widgets)
list(APPEND NDA_PLUGIN_TARGETS NadeTextSinkPlugin)

message(STATUS "Nade text messaging plugins enabled: NadeTextSourcePlugin, NadeEncryptorPlugin, NadeDecryptorPlugin, NadeTextSinkPlugin")

# ============================================================================

# Platform-specific settings
if(WIN32)
    # Windows audio APIs
    target_compile_definitions(${PROJECT_NAME} PRIVATE INITGUID)
    target_link_libraries(${PROJECT_NAME} winmm dsound ole32 propsys)
    # Enable console for debugging
    set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE OFF)
endif()

# Install targets
install(TARGETS ${PROJECT_NAME} DESTINATION bin)
if(NDA_PLUGIN_TARGETS)
    install(TARGETS ${NDA_PLUGIN_TARGETS}
        RUNTIME DESTINATION plugins
        LIBRARY DESTINATION plugins
        ARCHIVE DESTINATION plugins)
endif()
