using System;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Alchroma
{
    /// <summary>
    /// Identity of this install as the backend sees it. The id is a GUID created
    /// once per install (more stable than SystemInfo.deviceUniqueIdentifier, which
    /// needs extra permissions on some Android versions).
    /// </summary>
    public static class AlchromaDevice
    {
        const string IdKey = "alchroma.deviceId";
        static string id;

        /// <summary>Set this before the first backend call to override the auto-detected type ("headset", "tablet", "phone", "projector").</summary>
        public static string ForceType;

        public static string Id
        {
            get
            {
                if (string.IsNullOrEmpty(id))
                {
                    id = PlayerPrefs.GetString(IdKey, "");
                    if (string.IsNullOrEmpty(id))
                    {
                        id = Guid.NewGuid().ToString("N");
                        PlayerPrefs.SetString(IdKey, id);
                        PlayerPrefs.Save();
                    }
                }
                return id;
            }
        }

        public static bool IsHeadset
        {
            get
            {
                var model = SystemInfo.deviceModel ?? "";
                return model.IndexOf("Quest", StringComparison.OrdinalIgnoreCase) >= 0
                    || model.IndexOf("Oculus", StringComparison.OrdinalIgnoreCase) >= 0
                    || model.IndexOf("Pico", StringComparison.OrdinalIgnoreCase) >= 0;
            }
        }

        public static bool IsTablet
        {
            get
            {
                if (Screen.dpi <= 0) return false;
                float w = Screen.width / Screen.dpi, h = Screen.height / Screen.dpi;
                return Mathf.Sqrt(w * w + h * h) >= 6.5f;
            }
        }

        public static string Type
        {
            get
            {
                if (!string.IsNullOrEmpty(ForceType)) return ForceType;
                if (IsHeadset) return "headset";
                return IsTablet ? "tablet" : "phone";
            }
        }

        public static string Platform
        {
            get
            {
#if UNITY_IOS
                return "ios";
#elif UNITY_ANDROID
                return IsHeadset ? "quest" : "android";
#elif UNITY_STANDALONE_WIN
                return "windows";
#elif UNITY_STANDALONE_OSX
                return "macos";
#else
                return "other";
#endif
            }
        }

        public static string Name => string.IsNullOrEmpty(SystemInfo.deviceModel) ? Type : SystemInfo.deviceModel;

        /// <summary>Body for /api/devices/register, /pair/start and /pair/redeem.</summary>
        public static JObject Info()
        {
            return new JObject
            {
                ["device_id"] = Id,
                ["device_type"] = Type,
                ["platform"] = Platform,
                ["name"] = Name,
                ["app_version"] = Application.version,
            };
        }
    }
}
