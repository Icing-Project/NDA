# CMake toolchain file for cross-compiling to Windows using MinGW-w64

# Target system
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

# Cross-compiler
set(CMAKE_C_COMPILER /usr/bin/x86_64-w64-mingw32-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/x86_64-w64-mingw32-g++)
set(CMAKE_RC_COMPILER /usr/bin/x86_64-w64-mingw32-windres)

# Target environment
set(CMAKE_FIND_ROOT_PATH /usr/x86_64-w64-mingw32/sys-root/mingw)

# Search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Search for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Qt6 specific settings
set(QT_MOC_EXECUTABLE /usr/bin/x86_64-w64-mingw32-moc-qt6)
set(QT_RCC_EXECUTABLE /usr/bin/x86_64-w64-mingw32-rcc-qt6)
set(QT_UIC_EXECUTABLE /usr/bin/x86_64-w64-mingw32-uic-qt6)

# Tell CMake where to find Qt6
set(Qt6_DIR /usr/x86_64-w64-mingw32/sys-root/mingw/lib/cmake/Qt6)
set(CMAKE_PREFIX_PATH
    /usr/x86_64-w64-mingw32/sys-root/mingw
    /usr/x86_64-w64-mingw32/sys-root/mingw/lib/cmake
)

# OpenSSL paths
set(OPENSSL_ROOT_DIR /usr/x86_64-w64-mingw32/sys-root/mingw)
set(OPENSSL_INCLUDE_DIR /usr/x86_64-w64-mingw32/sys-root/mingw/include)
set(OPENSSL_CRYPTO_LIBRARY /usr/x86_64-w64-mingw32/sys-root/mingw/lib/libcrypto.dll.a)
set(OPENSSL_SSL_LIBRARY /usr/x86_64-w64-mingw32/sys-root/mingw/lib/libssl.dll.a)

# Python paths
set(Python3_ROOT_DIR /usr/x86_64-w64-mingw32/sys-root/mingw)
set(Python3_INCLUDE_DIR /usr/x86_64-w64-mingw32/sys-root/mingw/include/python3.11)
set(Python3_LIBRARY /usr/x86_64-w64-mingw32/sys-root/mingw/lib/libpython3.11.dll.a)

# Additional compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libgcc -static-libstdc++")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
