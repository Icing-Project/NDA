{
  "targets": [
    {
      "target_name": "windows_audio",
      "sources": [
        "windows-audio.cpp",
        "audio-engine.cpp",
        "wasapi-wrapper.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")"
      ],
      "conditions": [
        ["OS=='win'", {
          "libraries": [
            "-lole32",
            "-loleaut32",
            "-luuid",
            "-lwinmm",
            "-lksuser"
          ],
          "msvs_settings": {
            "VCCLCompilerTool": {
              "ExceptionHandling": 1,
              "AdditionalOptions": [ "/EHsc" ]
            }
          }
        }]
      ],
      "defines": [ "NAPI_CPP_EXCEPTIONS" ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ]
    }
  ]
}