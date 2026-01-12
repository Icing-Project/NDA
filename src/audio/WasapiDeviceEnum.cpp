#include "audio/WasapiDeviceEnum.h"

#ifdef _WIN32
#include <windows.h>
#include <mmdeviceapi.h>
#include <functiondiscoverykeys_devpkey.h>
#endif

namespace nda {

#ifdef _WIN32
// Helper to convert wstring to UTF-8 string
static std::string wstringToUtf8(const std::wstring& wstr)
{
    if (wstr.empty()) return std::string();
    int sizeNeeded = WideCharToMultiByte(CP_UTF8, 0, wstr.data(),
                                          static_cast<int>(wstr.size()),
                                          nullptr, 0, nullptr, nullptr);
    if (sizeNeeded <= 0) return std::string();
    std::string result(sizeNeeded, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(),
                        static_cast<int>(wstr.size()),
                        &result[0], sizeNeeded, nullptr, nullptr);
    return result;
}
#endif

std::vector<WASAPIDeviceInfo> enumerateWASAPIDevices(int direction)
{
    std::vector<WASAPIDeviceInfo> devices;

#ifdef _WIN32
    // direction: 0 = eCapture (microphones), 1 = eRender (speakers)
    EDataFlow dataFlow = static_cast<EDataFlow>((direction == 0) ? eCapture : eRender);

    // Initialize COM (may already be initialized by Qt, that's OK)
    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool comInitialized = SUCCEEDED(hr);

    // Create device enumerator
    IMMDeviceEnumerator* enumerator = nullptr;
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr,
                          CLSCTX_ALL, __uuidof(IMMDeviceEnumerator),
                          (void**)&enumerator);
    if (FAILED(hr)) {
        if (comInitialized) CoUninitialize();
        return devices;
    }

    // Enumerate active audio endpoints
    IMMDeviceCollection* collection = nullptr;
    hr = enumerator->EnumAudioEndpoints(dataFlow, DEVICE_STATE_ACTIVE, &collection);
    if (FAILED(hr)) {
        enumerator->Release();
        if (comInitialized) CoUninitialize();
        return devices;
    }

    UINT count = 0;
    collection->GetCount(&count);

    for (UINT i = 0; i < count; ++i) {
        IMMDevice* device = nullptr;
        if (FAILED(collection->Item(i, &device))) continue;

        // Get device ID (GUID string)
        LPWSTR pwszID = nullptr;
        device->GetId(&pwszID);
        if (pwszID) {
            std::wstring wid(pwszID);
            std::string id = wstringToUtf8(wid);
            CoTaskMemFree(pwszID);

            // Get friendly name from property store
            IPropertyStore* props = nullptr;
            std::string friendlyName = "Unknown Device";
            if (SUCCEEDED(device->OpenPropertyStore(STGM_READ, &props))) {
                PROPVARIANT varName;
                PropVariantInit(&varName);
                if (SUCCEEDED(props->GetValue(PKEY_Device_FriendlyName, &varName))) {
                    if (varName.pwszVal) {
                        std::wstring wname(varName.pwszVal);
                        friendlyName = wstringToUtf8(wname);
                    }
                    PropVariantClear(&varName);
                }
                props->Release();
            }

            WASAPIDeviceInfo info;
            info.id = id;
            info.friendlyName = friendlyName;
            devices.push_back(info);
        }

        device->Release();
    }

    collection->Release();
    enumerator->Release();
    if (comInitialized) CoUninitialize();
#endif

    return devices;
}

} // namespace nda
