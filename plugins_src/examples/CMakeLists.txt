# Example Processor Plugins for NDA v2.0
# These demonstrate the AudioProcessorPlugin interface

cmake_minimum_required(VERSION 3.16)

# AES256 Encryptor Plugin
add_library(AES256EncryptorPlugin SHARED
    AES256EncryptorPlugin.cpp
)

target_include_directories(AES256EncryptorPlugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(AES256EncryptorPlugin
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(WIN32)
    target_compile_definitions(AES256EncryptorPlugin PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

set_target_properties(AES256EncryptorPlugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
)

# AES256 Decryptor Plugin
add_library(AES256DecryptorPlugin SHARED
    AES256DecryptorPlugin.cpp
)

target_include_directories(AES256DecryptorPlugin PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(AES256DecryptorPlugin
    OpenSSL::SSL
    OpenSSL::Crypto
)

if(WIN32)
    target_compile_definitions(AES256DecryptorPlugin PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
endif()

set_target_properties(AES256DecryptorPlugin PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    LIBRARY_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
    ARCHIVE_OUTPUT_DIRECTORY ${PLUGIN_OUTPUT_DIR}
)

# Add to plugin targets for installation
list(APPEND NDA_PLUGIN_TARGETS AES256EncryptorPlugin AES256DecryptorPlugin)
set(NDA_PLUGIN_TARGETS ${NDA_PLUGIN_TARGETS} PARENT_SCOPE)

message(STATUS "Example processor plugins configured (AES256 Encryptor/Decryptor)")

