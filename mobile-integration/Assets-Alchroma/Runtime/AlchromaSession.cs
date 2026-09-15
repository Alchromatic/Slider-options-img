using System;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Alchroma
{
    [Serializable]
    public class AlchromaUser
    {
        public string id;
        public string email;
        public string name;
        public string organization_name;
        public string workspace_id;
    }

    /// <summary>
    /// The signed-in account on this device, persisted in PlayerPrefs.
    /// TokenType is "user" right after an email/password login and "device" once
    /// the device has been registered/paired (long-lived, revocable from the
    /// webapp Devices page).
    /// </summary>
    public static class AlchromaSession
    {
        const string Key = "alchroma.session";

        public static string AccessToken { get; private set; }
        public static string TokenType { get; private set; }
        public static AlchromaUser User { get; private set; }

        public static bool IsLoggedIn => !string.IsNullOrEmpty(AccessToken) && User != null;

        /// <summary>Raised after Set() and Clear().</summary>
        public static event Action Changed;

        static bool loaded;

        public static void Load()
        {
            if (loaded) return;
            loaded = true;
            try
            {
                var raw = PlayerPrefs.GetString(Key, "");
                if (string.IsNullOrEmpty(raw)) return;
                var j = JObject.Parse(raw);
                AccessToken = (string)j["access_token"];
                TokenType = (string)j["token_type"] ?? "user";
                User = j["user"] != null && j["user"].Type == JTokenType.Object ? j["user"].ToObject<AlchromaUser>() : null;
            }
            catch (Exception e)
            {
                Debug.LogWarning("[Alchroma] Stored session could not be read, clearing it: " + e.Message);
                Clear();
            }
        }

        public static void Set(string accessToken, string tokenType, AlchromaUser user)
        {
            loaded = true;
            AccessToken = accessToken;
            TokenType = string.IsNullOrEmpty(tokenType) ? "user" : tokenType;
            User = user;
            var j = new JObject
            {
                ["access_token"] = accessToken,
                ["token_type"] = TokenType,
                ["user"] = user == null ? null : JObject.FromObject(user),
            };
            PlayerPrefs.SetString(Key, j.ToString(Formatting.None));
            PlayerPrefs.Save();
            Changed?.Invoke();
        }

        public static void Clear()
        {
            loaded = true;
            AccessToken = null;
            TokenType = null;
            User = null;
            PlayerPrefs.DeleteKey(Key);
            PlayerPrefs.Save();
            Changed?.Invoke();
        }
    }
}
