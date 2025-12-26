cmake_minimum_required(VERSION 3.16)
project(NDA VERSION 1.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# Option to enable Python plugin support
option(NDA_ENABLE_PYTHON "Enable Python plugin support" ON)

# Qt6 packages
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui Network)

# OpenSSL for encryption (now plugin-only, not core dependency in v2.0)
find_package(OpenSSL REQUIRED)

# Optional: libsamplerate for high-quality resampling
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
else()
    message(STATUS "pkg-config not found - libsamplerate detection skipped")
endif()

# Find Python and NumPy if Python support is enabled
if(NDA_ENABLE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development NumPy)
    if(Python3_FOUND)
        message(STATUS "Python support enabled")
        message(STATUS "  Python version: ${Python3_VERSION}")
        message(STATUS "  Python include dirs: ${Python3_INCLUDE_DIRS}")
        message(STATUS "  NumPy include dirs: ${Python3_NumPy_INCLUDE_DIRS}")
        add_compile_definitions(NDA_ENABLE_PYTHON)
    else()
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
    src/ui/AudioDevicesView.cpp
    src/ui/EncryptionView.cpp
    src/ui/PluginsView.cpp
    src/ui/SettingsView.cpp
    # DELETED: Dashboard.cpp, PipelineView.cpp (merged into UnifiedPipelineView)
    src/audio/AudioEngine.cpp
    src/audio/AudioDevice.cpp
    src/audio/Resampler.cpp
    src/core/ProcessingPipeline.cpp
    src/plugins/PluginManager.cpp
)

# Add Python bridge if enabled
if(NDA_ENABLE_PYTHON)
    list(APPEND SOURCES src/plugins/PythonPluginBridge.cpp)
endif()

# Header files (v2.0: Unified view)
set(HEADERS
    include/ui/MainWindow.h
    include/ui/UnifiedPipelineView.h
    include/ui/PluginSidebar.h
    include/ui/AudioDevicesView.h
    include/ui/EncryptionView.h
    include/ui/PluginsView.h
    include/ui/SettingsView.h
    # DELETED: Dashboard.h, PipelineView.h (merged into UnifiedPipelineView)
    include/audio/AudioEngine.h
    include/audio/AudioDevice.h
    include/audio/AudioBuffer.h
    include/audio/Resampler.h
    include/core/ProcessingPipeline.h
    include/plugins/PluginTypes.h
    include/plugins/BasePlugin.h
    include/plugins/AudioSourcePlugin.h
    include/plugins/AudioProcessorPlugin.h
    include/plugins/AudioSinkPlugin.h
    include/plugins/PluginManager.h
)

# Add Python bridge header if enabled
if(NDA_ENABLE_PYTHON)
    list(APPEND HEADERS include/plugins/PythonPluginBridge.h)
endif()

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    Qt6::Core
    Qt6::Widgets
    Qt6::Gui
    Qt6::Network
    OpenSSL::SSL
    OpenSSL::Crypto
)

# Link libsamplerate if available
if(SAMPLERATE_FOUND)
    target_link_libraries(${PROJECT_NAME} ${SAMPLERATE_LIBRARIES})
endif()

# Link Python libraries if enabled
if(NDA_ENABLE_PYTHON)
    target_include_directories(${PROJECT_NAME} PRIVATE
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )
    target_link_libraries(${PROJECT_NAME} Python3::Python)
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
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
        LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
        ARCHIVE_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    )
endfunction()

set(NDA_PLUGIN_TARGETS)

if(WIN32)
    add_nda_plugin(AIOCSourcePlugin
        plugins_src/AIOCPluginCommon.cpp
        plugins_src/AIOCSourcePlugin.cpp
    )
    target_link_libraries(AIOCSourcePlugin ole32 avrt setupapi hid)
    list(APPEND NDA_PLUGIN_TARGETS AIOCSourcePlugin)

    add_nda_plugin(AIOCSinkPlugin
        plugins_src/AIOCPluginCommon.cpp
        plugins_src/AIOCSinkPlugin.cpp
    )
    target_link_libraries(AIOCSinkPlugin ole32 avrt setupapi hid)
    list(APPEND NDA_PLUGIN_TARGETS AIOCSinkPlugin)
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

# Add example processor plugins (AES256 Encryptor/Decryptor)
# add_subdirectory(plugins_src/examples)

# Platform-specific settings
if(WIN32)
    # Windows audio APIs
    target_link_libraries(${PROJECT_NAME} winmm dsound)
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
