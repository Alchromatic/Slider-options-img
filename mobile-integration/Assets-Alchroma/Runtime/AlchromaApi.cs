using System;
using System.Collections;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Networking;

namespace Alchroma
{
    /// <summary>Result of one backend call. `json` is null when the body was not a JSON object.</summary>
    public class ApiResponse
    {
        public bool ok;
        public long status;
        public string text;
        public JObject json;
        public string error;

        /// <summary>The backend's `detail` message when there is one, else the transport error.</summary>
        public string Detail
        {
            get
            {
                var d = json?["detail"];
                if (d != null && d.Type == JTokenType.String) return (string)d;
                if (d != null && d.Type == JTokenType.Array && d.HasValues) return d.First?["msg"]?.ToString() ?? d.ToString(Formatting.None);
                return error;
            }
        }

        public T As<T>() => json == null ? default : json.ToObject<T>();
    }

    /// <summary>
    /// Thin JSON client for the GeoMagic backend. Lives on a persistent
    /// "[Alchroma]" GameObject that is created on first use. Adds the Bearer
    /// token automatically and clears the session when the backend says the
    /// device was revoked or the token expired.
    /// </summary>
    public class AlchromaApi : MonoBehaviour
    {
        static AlchromaApi instance;

        public static AlchromaApi Instance
        {
            get
            {
                if (instance == null)
                {
                    var go = new GameObject("[Alchroma]");
                    DontDestroyOnLoad(go);
                    instance = go.AddComponent<AlchromaApi>();
                }
                return instance;
            }
        }

        /// <summary>Raised when a request came back 401 for a session we thought was valid.</summary>
        public static event Action SessionLost;

        public int timeoutSeconds = 30;

        void Awake()
        {
            if (instance != null && instance != this)
            {
                Destroy(this);
                return;
            }
            instance = this;
            DontDestroyOnLoad(gameObject);
            AlchromaSession.Load();
        }

        public static void Get(string path, Action<ApiResponse> done, bool auth = true)
            => Instance.StartCoroutine(Instance.Send("GET", path, null, done, auth));

        public static void Post(string path, object body, Action<ApiResponse> done, bool auth = true)
            => Instance.StartCoroutine(Instance.Send("POST", path, body, done, auth));

        public static void Patch(string path, object body, Action<ApiResponse> done, bool auth = true)
            => Instance.StartCoroutine(Instance.Send("PATCH", path, body, done, auth));

        public static void Delete(string path, Action<ApiResponse> done, bool auth = true)
            => Instance.StartCoroutine(Instance.Send("DELETE", path, null, done, auth));

        IEnumerator Send(string method, string path, object body, Action<ApiResponse> done, bool auth)
        {
            var url = AlchromaConfig.Url(path);
            using (var req = new UnityWebRequest(url, method))
            {
                if (body != null)
                {
                    var json = body as string ?? JsonConvert.SerializeObject(body);
                    req.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
                    req.SetRequestHeader("Content-Type", "application/json");
                }
                req.downloadHandler = new DownloadHandlerBuffer();
                req.SetRequestHeader("Accept", "application/json");
                req.timeout = timeoutSeconds;
                if (auth && !string.IsNullOrEmpty(AlchromaSession.AccessToken))
                    req.SetRequestHeader("Authorization", "Bearer " + AlchromaSession.AccessToken);

                yield return req.SendWebRequest();

                var res = new ApiResponse { status = req.responseCode, text = req.downloadHandler != null ? req.downloadHandler.text : null };
                try
                {
                    if (!string.IsNullOrWhiteSpace(res.text) && res.text.TrimStart().StartsWith("{"))
                        res.json = JObject.Parse(res.text);
                }
                catch (Exception) { /* not a JSON object body */ }

                res.ok = req.result == UnityWebRequest.Result.Success;
                if (!res.ok)
                {
                    if (req.result == UnityWebRequest.Result.ConnectionError)
                        res.error = "Network error: " + req.error;
                    else
                        res.error = res.Detail ?? ("HTTP " + req.responseCode + " " + req.error);
                }

                if (AlchromaConfig.VerboseLogs)
                    Debug.Log($"[Alchroma] {method} {path} -> {res.status}{(res.ok ? "" : " " + res.error)}");

                if (res.status == 401 && auth && AlchromaSession.IsLoggedIn)
                {
                    var detail = res.Detail ?? "";
                    if (detail == "device_revoked" || detail.StartsWith("Invalid or expired"))
                    {
                        Debug.LogWarning("[Alchroma] Session is no longer valid (" + detail + "), signing out.");
                        AlchromaSession.Clear();
                        SessionLost?.Invoke();
                    }
                }

                done?.Invoke(res);
            }
        }
    }
}
