using UnityEngine;

namespace Alchroma
{
    /// <summary>
    /// Where the GeoMagic / Alchroma backend lives and a few global switches.
    /// The base URL is persisted so a debug menu can point a build at staging
    /// without a rebuild (AlchromaConfig.BaseUrl = "http://192.168.1.10:8000").
    /// </summary>
    public static class AlchromaConfig
    {
        public const string DefaultBaseUrl = "https://alchromaticdemo.up.railway.app";

        /// <summary>Custom URL scheme handled by the apps (alchroma://pair?code=XXXX).</summary>
        public const string DeepLinkScheme = "alchroma";

        const string BaseUrlKey = "alchroma.baseUrl";

        public static string BaseUrl
        {
            get => PlayerPrefs.GetString(BaseUrlKey, DefaultBaseUrl).TrimEnd('/');
            set
            {
                PlayerPrefs.SetString(BaseUrlKey, string.IsNullOrEmpty(value) ? DefaultBaseUrl : value.TrimEnd('/'));
                PlayerPrefs.Save();
            }
        }

        public static string Url(string path)
        {
            if (string.IsNullOrEmpty(path)) return BaseUrl;
            return BaseUrl + (path.StartsWith("/") ? path : "/" + path);
        }

        /// <summary>
        /// When true (default) the SDK shows its own sign-in / pairing overlay at
        /// startup whenever nobody is signed in. Set it to false from a
        /// [RuntimeInitializeOnLoadMethod(BeforeSceneLoad)] in your own code if you
        /// want to drive AlchromaLoginUI (or your own UI) yourself.
        /// </summary>
        public static bool AutoShowLogin = true;

        /// <summary>Seconds between heartbeats that keep the device "online" on the Devices page.</summary>
        public static float HeartbeatSeconds = 60f;

        public static bool VerboseLogs = true;
    }
}
