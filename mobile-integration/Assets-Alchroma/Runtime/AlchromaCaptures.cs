using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace Alchroma
{
    /// <summary>
    /// Colors captured on this device. Every Save() lands in the user's account
    /// (Devices page + "My Colors" in the Paint Collection). When nobody is signed
    /// in or the network is down the capture is queued in PlayerPrefs and flushed
    /// by FlushQueue() (called at startup and after every successful sign-in).
    /// </summary>
    public static class AlchromaCaptures
    {
        [Serializable]
        public class RecipeItem
        {
            public string name;
            public string hex;
            public float? percentage;
            public float? parts;
        }

        [Serializable]
        public class Capture
        {
            public int id;
            public string hex;
            public string rgb;          // "r,g,b"
            public string name;
            public string source;       // camera | unmix | picker | projector | web_camera | manual | other
            public string device_id;
            public string device_type;
            public string device_name;
            public List<RecipeItem> recipe;
            public string created_at;
        }

        const string QueueKey = "alchroma.captureQueue";
        const int QueueCap = 500;

        /// <summary>done(success, capture, error). error == "not_logged_in" means it was queued.</summary>
        public static void Save(string hex, string name = null, string source = "camera",
            IList<RecipeItem> recipe = null, JObject meta = null, Action<bool, Capture, string> done = null)
        {
            var clean = NormalizeHex(hex);
            if (clean == null)
            {
                done?.Invoke(false, null, "Invalid color: " + hex);
                return;
            }
            var body = new JObject
            {
                ["hex"] = clean,
                ["name"] = string.IsNullOrWhiteSpace(name) ? null : name.Trim(),
                ["source"] = string.IsNullOrEmpty(source) ? "camera" : source,
                ["device_id"] = AlchromaDevice.Id,
            };
            if (recipe != null && recipe.Count > 0) body["recipe"] = JArray.FromObject(recipe);
            if (meta != null) body["meta"] = meta;

            if (!AlchromaSession.IsLoggedIn)
            {
                Enqueue(body);
                done?.Invoke(false, null, "not_logged_in");
                return;
            }
            AlchromaApi.Post("/api/devices/captures", body, res =>
            {
                if (!res.ok)
                {
                    // keep it for later if the server was unreachable, drop it if it was rejected
                    if (res.status == 0 || res.status >= 500) Enqueue(body);
                    done?.Invoke(false, null, res.Detail ?? "Save failed");
                    return;
                }
                done?.Invoke(true, res.json?["capture"]?.ToObject<Capture>(), null);
            });
        }

        /// <summary>Newest first. done(success, captures, total)</summary>
        public static void List(int limit, Action<bool, List<Capture>, int> done, string source = null)
        {
            if (!AlchromaSession.IsLoggedIn)
            {
                done?.Invoke(false, new List<Capture>(), 0);
                return;
            }
            var path = "/api/devices/captures?limit=" + Mathf.Clamp(limit, 1, 500);
            if (!string.IsNullOrEmpty(source)) path += "&source=" + Uri.EscapeDataString(source);
            AlchromaApi.Get(path, res =>
            {
                if (!res.ok || res.json == null)
                {
                    done?.Invoke(false, new List<Capture>(), 0);
                    return;
                }
                var list = res.json["captures"]?.ToObject<List<Capture>>() ?? new List<Capture>();
                done?.Invoke(true, list, (int?)res.json["total"] ?? list.Count);
            });
        }

        public static void Delete(int id, Action<bool, string> done)
        {
            AlchromaApi.Delete("/api/devices/captures/" + id, res => done?.Invoke(res.ok, res.ok ? null : res.Detail));
        }

        // ------------------------------------------------------------- queue

        public static int QueuedCount => LoadQueue().Count;

        public static void FlushQueue()
        {
            if (!AlchromaSession.IsLoggedIn) return;
            var q = LoadQueue();
            if (q.Count == 0) return;
            var batch = new JArray();
            for (int i = 0; i < Mathf.Min(q.Count, 200); i++) batch.Add(q[i]);
            AlchromaApi.Post("/api/devices/captures/batch", new JObject { ["captures"] = batch }, res =>
            {
                if (!res.ok) return;
                var rest = new JArray();
                for (int i = batch.Count; i < q.Count; i++) rest.Add(q[i]);
                SaveQueue(rest);
                if (AlchromaConfig.VerboseLogs) Debug.Log("[Alchroma] Flushed " + batch.Count + " queued capture(s)");
                if (rest.Count > 0) FlushQueue();
            });
        }

        static void Enqueue(JObject body)
        {
            var q = LoadQueue();
            q.Add(body);
            while (q.Count > QueueCap) q.RemoveAt(0);
            SaveQueue(q);
        }

        static JArray LoadQueue()
        {
            try
            {
                var raw = PlayerPrefs.GetString(QueueKey, "");
                return string.IsNullOrEmpty(raw) ? new JArray() : JArray.Parse(raw);
            }
            catch (Exception)
            {
                return new JArray();
            }
        }

        static void SaveQueue(JArray q)
        {
            PlayerPrefs.SetString(QueueKey, q.ToString(Newtonsoft.Json.Formatting.None));
            PlayerPrefs.Save();
        }

        /// <summary>"1e2448", "#1E2448" or "1E2448FF" -> "#1E2448"; null when it is not a color.</summary>
        public static string NormalizeHex(string hex)
        {
            var h = (hex ?? "").Trim().TrimStart('#').ToUpperInvariant();
            if (h.Length == 8) h = h.Substring(0, 6);
            if (h.Length == 3) h = "" + h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
            if (h.Length != 6) return null;
            foreach (var c in h)
                if (!Uri.IsHexDigit(c)) return null;
            return "#" + h;
        }

        public static string ColorToHex(Color color)
        {
            Color32 c = color;
            return string.Format("#{0:X2}{1:X2}{2:X2}", c.r, c.g, c.b);
        }
    }
}
